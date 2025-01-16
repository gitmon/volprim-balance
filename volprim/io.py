# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
I/O routines for handling Mitsuba scene assets. Includes functions to load
"Python assets" (Mitsuba scenes stored as Python modules using dictionaries)
and convert existing scenes into dictionaries, which can be written to disk
as Python assets.
'''

import os, shutil, logging
from os.path import join, basename, splitext, exists
import numpy as np

import drjit as dr
import mitsuba as mi

def get_plugin_type(obj):
    '''
    Fetch the plugin type name for a given instance.
    This function also handle the corner cases of expanded objects
    '''
    alias = obj.class_().alias()
    if alias == 'scene':
        return 'scene'
    plugin_type = mi.PluginManager.instance().get_plugin_type(alias)
    if alias == 'CppADIntegrator':
        plugin_type = {
            "<class 'volprim.integrators.volprim_rf.VolumetricPrimitiveRadianceFieldIntegrator'>": 'volprim_rf',
            "<class 'volprim.integrators.volprim_tomography.VolumetricPrimitiveTomographyIntegrator'>": 'volprim_tomography',
            "<class 'volprim.integrators.volprim_prb.VolumetricPrimitivesPRBIntegrator'>": 'volprim_prb',
            "<class 'mitsuba.ad.integrators.prb.PRBIntegrator'>": 'prb',
        }[str(type(obj))]
    elif plugin_type == '':
        plugin_type = {
            'BitmapTextureImpl_FP16': 'bitmap',
            'BitmapTextureImpl': 'bitmap',
        }[alias]
    return plugin_type

def asset_to_dict(asset,
                  objects=True,
                  emitters=True,
                  sensors=True,
                  integrator=True) -> dict:
    '''
    Assemble a scene Python dictionary for a given asset

    Parameter:
        asset: (str or module) path to asset or Python module
    '''
    if isinstance(asset, str):
        from importlib.machinery import SourceFileLoader
        init_path = join(asset, '__init__.py')
        if not exists(init_path):
            raise Exception(f'Invalid asset path: {init_path}')
        asset = SourceFileLoader("asset", init_path).load_module()

    d = { 'type': 'scene' }
    if objects:
        d.update(getattr(asset, 'OBJECTS',  {}))
    if emitters:
        d.update(getattr(asset, 'EMITTERS', {}))
    if sensors:
        d.update(getattr(asset, 'SENSORS',  {}))
    if integrator and hasattr(asset, 'INTEGRATOR'):
        d['integrator'] = asset.INTEGRATOR
    return d

def scale_films(d: dict, scale: float=1.0) -> dict:
    '''
    Scale the films resolution in the given scene dictionary
    '''
    def traverse(d):
        for k, v in d.items():
            if k == 'film':
                v['width']  = int(scale * v['width'])
                v['height'] = int(scale * v['height'])
            elif isinstance(v, dict):
                traverse(v)
    traverse(d)

    return d

def dict_to_asset(scene_dict: dict, output_folder: str, verbose=False):
    '''
    Generate a Python asset that contains a dictionary that represents a Mitsuba
    scene.
    '''
    sensor_types  = ['perspective', 'orthographic', 'thinlens']
    emitter_types = ['envmap', 'constant', 'point', 'distant', 'spot', 'directional']

    print(f'Writing asset to {output_folder} ...')

    def dict_to_string(d, output_folder, path, indent=0, add_resources=False) -> str:
        '''
        Convert a Python dictionary to its string Python code representation.
        The input dictionary will only contain strings as keys, and float, int, bool
        and stings as values.
        '''
        w        = lambda x: str(''.join([' ']) * indent) + x
        sanitize = lambda x: x.replace('.', '_')
        resolve  = lambda x: str(mi.Thread.thread().file_resolver().resolve(x))

        object_type = d.get('type', None)

        s = '{\n'
        if add_resources:
            s += w("    'resources': { 'type': 'resources', 'path': dirname(__file__) },\n")

        if object_type == 'meshholder':
            object_type = d['type'] = 'ply'
            m = mi.Mesh("", len(d['vertex_positions']) // 3, len(d['faces']) // 3)
            p = mi.traverse(m)
            p['faces']            = d['faces']
            p['vertex_positions'] = d['vertex_positions']
            p['vertex_normals']   = d['vertex_normals']
            p['vertex_texcoords'] = d['vertex_texcoords']
            p.update()
            os.makedirs(join(output_folder, 'meshes'), exist_ok=True)
            d['filename'] = join(output_folder, 'meshes', f'{path}.ply')
            m.write_ply(d['filename'])
            for k in ['faces', 'vertex_positions', 'vertex_normals', 'vertex_texcoords']:
                del d[k]

        is_ellipsoid = object_type and 'ellipsoid' in object_type
        if is_ellipsoid and not 'filename' in d:
            extras_keys = list(filter(lambda k: isinstance(d[k], np.ndarray), list(d.keys())))
            extras_keys = list(filter(lambda k: k not in ['centers', 'scales', 'quaternions'], extras_keys))

            os.makedirs(join(output_folder, 'data'), exist_ok=True)
            filename = join(output_folder, 'data', f'{path}.ply')
            ellipsoid_dict_to_ply(d, extras_keys, filename)

            for k in extras_keys + ['centers', 'scales', 'quaternions']:
                del d[k]
            d['filename'] = filename

        for k, v in d.items():
            if isinstance(v, dict) and v['type'] == 'resources':
                mi.Thread.thread().file_resolver().append(v['path'])
                continue

            s += w(f"    '{sanitize(k)}': ")

            if isinstance(v, dict):
                s += dict_to_string(v, output_folder, f'{path}.{k}', indent + 4)
                s += f",\n"
            elif isinstance(v, str):
                if k == 'filename':
                    src = resolve(v)
                    base, ext = splitext(basename(src))
                    dst_folder = {
                        '.obj': 'meshes',
                        '.ply': 'data' if is_ellipsoid else 'meshes',
                        '.jpg': 'textures',
                        '.png': 'textures',
                        '.exr': 'textures',
                        '.json': 'data',
                    }[ext]
                    os.makedirs(join(output_folder, dst_folder), exist_ok=True)
                    v = join(dst_folder, f'{base}{ext}')
                    dst = join(output_folder, v)
                    if not exists(dst):
                        shutil.copy(src, dst)
                    s += f"r\'{v}\',\n"
                else:
                    if k == 'id':
                        v = sanitize(v)
                    s += f"\'{v}\',\n"
            elif k == 'to_world':
                if object_type == 'envmap':
                    T = mi.ScalarTransform4f(v)
                    R = dr.transform_decompose(v.matrix)[1]
                    rotations = dr.quat_to_euler(R)
                    if any([r != 0.0 for r in rotations]):
                        s += f"T()"
                        if rotations[0] != 0.0:
                            s += f".rotate([1, 0, 0], {rotations[0]})"
                        if rotations[1] != 0.0:
                            s += f".rotate([0, 1, 0], {rotations[1]})"
                        if rotations[2] != 0.0:
                            s += f".rotate([0, 0, 1], {rotations[2]})"
                        s += f"\n"
                elif object_type in sensor_types:
                    T = mi.ScalarTransform4f(v)
                    origin    = T @ mi.ScalarPoint3f(0, 0, 0)
                    target    = T @ mi.ScalarPoint3f(0, 0, 1)
                    up        = T @ mi.ScalarVector3f(0, 1, 0)
                    s += f"T().look_at(\n"
                    s += w(f"         origin={origin},\n")
                    s += w(f"         target={target},\n")
                    s += w(f"         up={up},\n")
                    s += w(f"     ),\n")
                else:
                    s += f"T([{v.matrix[0]}, {v.matrix[1]}, {v.matrix[2]}, {v.matrix[3]}]),\n"
            elif k == 'wrap_mode' or k == 'filter_mode':
                s += f"dr.{v},\n"
            elif isinstance(v, np.ndarray) or dr.is_tensor_v(v):
                os.makedirs(join(output_folder, 'data'), exist_ok=True)
                filename = f'data/{path}.{k}.npy'
                np.save(join(output_folder, filename), np.array(v))
                s += f"np.load(join(dirname(__file__), \'{filename}\')),\n"
            elif isinstance(v, list):
                s += f"{v},\n"
            elif isinstance(v, mi.ScalarTransform3f):
                s += f"mi.ScalarTransform3f([\n"
                for i in range(3):
                    s += w(f"         {v.matrix[i]},\n")
                s += w(f"     ]),\n")
            elif isinstance(v, mi.ScalarTransform4f):
                s += f"T([\n"
                for i in range(4):
                    s += w(f"         {v.matrix[i]},\n")
                s += w(f"     ]),\n")
            elif isinstance(v, mi.Bitmap):
                os.makedirs(join(output_folder, 'textures'), exist_ok=True)
                filename = join('textures', f'{path}.exr')
                mi.util.write_bitmap(join(output_folder, filename), v)
                s += f'r\'{filename}\',\n'
            else:
                s += f"{v},\n"
        s += w('}')

        s = s.replace('\'bitmap\': ', '\'filename\': ')
        s = s.replace('\\', '/')

        return s

    assert scene_dict['type'] == 'scene', 'can only process scene dictionary!'

    # Sort scene elements
    sensors  = {}
    emitters = {}
    objects  = {}
    for k, v in scene_dict.items():
        if isinstance(v, str) and v == 'scene':
            continue
        if v['type'] in sensor_types:
            sensors[k] = v
        elif v['type'] in emitter_types:
            emitters[k] = v
        else:
            objects[k] = v

    os.makedirs(output_folder, exist_ok=True)
    filename = join(output_folder, '__init__.py')
    with open(filename, 'w') as f:
        f.write('import os\n')
        f.write('from os.path import join, dirname\n')
        f.write('import numpy as np\n')
        f.write('import drjit as dr\n')
        f.write('import mitsuba as mi\n')
        f.write('from mitsuba.scalar_rgb import ScalarTransform4f as T\n')

        f.write('\n')

        f.write('OBJECTS = ')
        f.write(dict_to_string(objects, output_folder, 'root', add_resources=True))
        f.write('\n')
        f.write('\n')

        f.write('SENSORS = ')
        f.write(dict_to_string(sensors, output_folder, 'root', add_resources=True))
        f.write('\n')
        f.write('\n')

        f.write('EMITTERS = ')
        f.write(dict_to_string(emitters, output_folder, 'root', add_resources=True))
        f.write('\n')


def object_to_dict(root):
    '''
    Routine to convert a Mitsuba object into a dictionary that could be passed
    directly to `mi.load_dict()` to re-load the same object.
    '''
    class ObjectToDictTraversal(mi.TraversalCallback):
        def __init__(self, node=None, name=None):
            mi.TraversalCallback.__init__(self)
            self._node = node
            self._name = name
            self._dict = {}
            if node is not None:
                self._plugin_type = get_plugin_type(node)
                self._dict = { 'type': self._plugin_type }

        def put_parameter(self, name, value, flags, value_type=None):
            # Filter out some of the parameters
            if name in ['enabled', 'silhouette_sampling_weight', 'sampling_weight']:
                return

            if value_type is not None:
                value = mi.get_property(value, value_type, self._node)

            if isinstance(value, (mi.Float, mi.UInt32, mi.Color3f, mi.Spectrum, mi.Vector3f)) and dr.width(value) == 1:
                value = dr.slice(value)
            elif isinstance(value, mi.Transform3f):
                value = mi.ScalarTransform3f(dr.slice(value.matrix))
            elif isinstance(value, mi.Transform4f):
                value = mi.ScalarTransform4f(dr.slice(value.matrix))

            self._dict[name] = value

        def put_object(self, name, node, flags):
            # Recursively traverse other object to assemble dictionary
            cb = ObjectToDictTraversal(node, name)
            node.traverse(cb)

            plugin_type = cb._plugin_type
            parent_name = node.class_().parent().name()

            # Handle special cases
            d = cb._dict

            # Meshes
            if parent_name == 'Mesh' and 'faces' in d:
                d['type'] = 'meshholder'
            # Ellipsoids
            elif 'ellipsoid' in plugin_type:
                prim_count = dr.width(d['data'].array) // 10
                data = np.array(d['data']).reshape(prim_count, -1)
                d['centers']     = data[:, 0:3]
                d['scales']      = data[:, 3:6]
                d['quaternions'] = data[:, 6:10]
                del d['data']
                for k2, v2 in d.items():
                    if dr.is_array_v(v2):
                        d[k2] = np.array(v2).reshape(prim_count, -1)
            # Bitmap
            elif plugin_type == 'bitmap':
                d['bitmap'] = mi.Bitmap(d['data'])
                del d['data']
                d['wrap_mode']   = dr.WrapMode(d['wrap_mode'])
                d['filter_mode'] = dr.FilterMode(d['filter_mode'])
            # Film
            elif plugin_type == 'hdrfilm':
                d['width']  = d['size'][0]
                d['height'] = d['size'][1]
                for k in ['size', 'crop_size', 'crop_offset']:
                    del d[k]
            # sRGB
            elif plugin_type == 'srgb':
                d['type'] = 'rgb'
            # Sensors
            elif plugin_type == 'perspective' or plugin_type == 'orthogonal':
                d['shutter_close'] = d['shutter_open'] + d['shutter_open_time']
                del d['shutter_open_time']
                if plugin_type == 'perspective':
                    d['fov'] = d['x_fov']
                    d['fov_axis'] = 'x'
                    del d['x_fov']

            self._dict[name] = d

    cb = ObjectToDictTraversal()
    cb.put_object('root', root, None)
    return cb._dict['root']

def ellipsoid_dict_to_ply(d, extras_keys, filename):
    '''
    Export an dictionary representing an ellipsoid shape into a PLY file.
    '''
    import plyfile

    is_3dg = 'sh_coeffs' in extras_keys and 'opacities' in extras_keys
    extras = { k: d[k].shape[1] for k in extras_keys }

    centers     = np.array(d['centers'])
    scales      = np.array(np.log(np.maximum(d['scales'], 1e-6)))
    quaternions = np.array(d['quaternions'])[:, [3, 0, 1, 2]] # Reorder (i, j, k, r -> r, i, j, k)
    normals     = np.zeros_like(centers)

    if is_3dg:
        sh_coeffs = np.array(d['sh_coeffs'])
        f_dc   = sh_coeffs[:, :3]
        f_rest = sh_coeffs[:, 3:]

        if f_rest.shape[1] > 0:
            # Re-ordering the spherical harmonic coefficients!
            sh_n = (f_rest.shape[1] // 3) + 1
            col_mapping = sum([[(j*3+0-3, j-1+3-3), (j*3+1-3,j-1+sh_n+2-3), (j*3+2-3, j-1+2*sh_n+1-3)] for j in range(1, sh_n)], [])
            col_indices = [a for a, b in sorted(col_mapping, key=lambda x: x[1])]
            f_rest = f_rest[:, col_indices]

        opacities = np.clip(d['opacities'], 1e-8, 1.0 - 1e-8)
        opacities = np.array(np.log(opacities) -  np.log(1.0 - opacities))
        extras_data = [f_dc, f_rest, opacities]
    else:
        extras_data = [d[k] for k, dim in extras.items()]

    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    if is_3dg:
        attributes += ['f_dc_0', 'f_dc_1', 'f_dc_2']
        for i in range(extras['sh_coeffs'] - 3):
            attributes.append(f'f_rest_{i}')
        attributes += ['opacity']
    else:
        for k, dim in extras.items():
            for i in range(dim):
                attributes.append(f'{k}_{i}')
    attributes += ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']

    elements = np.empty(centers.shape[0], dtype=[(attr, 'f4') for attr in attributes])
    attributes = np.concatenate((centers, normals, *extras_data, scales, quaternions), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = plyfile.PlyElement.describe(elements, 'vertex')
    d['filename'] = filename
    plyfile.PlyData([el]).write(filename)

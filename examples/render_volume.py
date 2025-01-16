# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is a simple example of how to use Mitsuba to render a Volumetric Primitives asset.
"""
import os, argparse
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
import volprim

parser = argparse.ArgumentParser(description='Render volume')
parser.add_argument('--output', type=str, default='smoke.exr', help='Path to the result output folder')
parser.add_argument('--volume', type=str, default='resources/smoke.ply', help='Path to .ply volumetric primitive file')
parser.add_argument('--sigmat_scale', type=float, default=10.0, help='Scaling factor for the sigmat value')
args = parser.parse_args()

scene = mi.load_dict({
    'resources': { 'type': 'resources', 'path': os.path.dirname(__file__) },
    'type': 'scene',
    'integrator': {
        'type': 'volprim_prb',
        'max_depth': -1,
    },
    'primitives': {
        'type': 'ellipsoidsmesh',
        'filename': args.volume,
    },
    'envmap': {
        'type': 'envmap',
        'filename': '../resources/qwantani_dusk_2_1k.exr',
    },
    'cam': {
        'type': 'perspective',
        'near_clip': 0.009999999776482582,
        'far_clip': 10000.0,
        'shutter_open': 0.0,
        'film': {
            'type': 'hdrfilm',
            'width': 512,
            'height': 512,
        },
        'sampler': {
            'type': 'independent',
        },
        'principal_point_offset_x': 0.0,
        'principal_point_offset_y': 0.0,
        'to_world': mi.ScalarTransform4f().look_at(
             origin=[-3.98825, -0.306404, -1.74332e-07],
             target=[-2.99119, -0.229803, -1.30749e-07],
             up=[-0.076601, 0.997062, -3.34833e-09],
         ),
        'shutter_close': 0.0,
        'fov': 40.0,
        'fov_axis': 'x',
    }
})

# Scaling sigmat value
params = mi.traverse(scene)
params['primitives.sigma_t'] = params['primitives.sigma_t'] * args.sigmat_scale
params.update()

# Render the scene
with volprim.benchmark.single_run('Rendering'):
    img = mi.render(scene, spp=64)

# Write image to file on disk
print(f'Writing rendered image to {args.output}')
mi.util.write_bitmap(args.output, img)

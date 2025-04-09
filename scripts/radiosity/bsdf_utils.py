import drjit as dr
import mitsuba as mi
from drjit.auto import Float, UInt, Bool
import polyscope as ps
from scripts.radiosity.visualizer import plot_mesh_attributes


# ----------------------- Helpers -----------------------
### Helper functions from `principledhelpers.h`

def mulsign_neg(a, b):
    # mulsign_neg(x) should be equiv. to -mulsign(x)
    return -dr.mulsign(a, b)

def fnmadd(a, b, c):
    # `fnmadd(a,b,c)` == `-a * b + c`, which can be implemented as either
    # `fmadd(-a,b,c)` or `fmadd(a,-b,c)`
    return dr.fma(-a, b, c)

def calc_dist_params(anisotropic: Float, roughness: Float, has_anisotropic: bool) -> tuple[Float, Float]:
    roughness_2 = dr.square(roughness)
    if not(has_anisotropic):
        a = dr.maximum(0.001, roughness_2)
        return (a, a)
    aspect = dr.sqrt(1.0 - 0.9 * anisotropic)
    return ( dr.maximum(0.001, roughness_2 / aspect),
             dr.maximum(0.001, roughness_2 * aspect))

def mac_mic_compatibility(m: mi.Vector3f, wi: mi.Vector3f, wo: mi.Vector3f, cos_theta_i: Float, reflection: bool) -> Bool:
    # NOTE: elementwise AND (this syntax should be fine)
    if reflection:
        return (dr.dot(wi, dr.mulsign(m, cos_theta_i)) > 0.0) & \
            (dr.dot(wo, dr.mulsign(m, cos_theta_i)) > 0.0)
    else:
        return (dr.dot(wi, dr.mulsign(m, cos_theta_i)) > 0.0) & \
            (dr.dot(wo, mulsign_neg(m, cos_theta_i)) > 0.0)

def schlick_R0_eta(eta: Float) -> Float:
    return dr.square((eta - 1.0) / (eta + 1.0))

def schlick_weight(cos_i: Float) -> Float:
    m = dr.clip(1.0 - cos_i, 0.0, 1.0)
    return dr.square(dr.square(m)) * m

def calc_schlick(R0: mi.Color3f, cos_theta_i: Float, eta: Float) -> mi.Color3f:
    outside_mask = cos_theta_i >= 0.0
    rcp_eta = dr.rcp(eta)
    eta_it  = dr.select(outside_mask, eta, rcp_eta)
    eta_ti  = dr.select(outside_mask, rcp_eta, eta)

    cos_theta_t_sqr = fnmadd(
            fnmadd(cos_theta_i, cos_theta_i, 1.0), dr.square(eta_ti), 1.0)
    cos_theta_t = dr.safe_sqrt(cos_theta_t_sqr)
    return dr.select(
            eta_it > 1.0,
            dr.lerp(schlick_weight(dr.abs(cos_theta_i)), 1.0, R0),
            dr.lerp(schlick_weight(cos_theta_t), 1.0, R0))

def principled_fresnel(
        F_dielectric: Float, 
        metallic: Float,
        spec_tint: Float,
        base_color: mi.Color3f,
        lum: Float, cos_theta_i: Float,
        front_side: Bool,
        bsdf: Float, eta: Float,
        has_metallic: bool, has_spec_tint: bool) -> mi.Color3f:
    # Outside mask based on micro surface
    outside_mask = cos_theta_i >= 0.0
    rcp_eta = dr.rcp(eta)
    eta_it  = dr.select(outside_mask, eta, rcp_eta)
    # TODO: does the width need to be set?
    F_schlick = mi.Color3f(0.0) 

    # Metallic component based on Schlick.
    if has_metallic:
        F_schlick += metallic * calc_schlick(base_color, cos_theta_i, eta)

    # Tinted dielectric component based on Schlick.
    if has_spec_tint:
        c_tint = dr.select(lum > 0.0, base_color / lum, 1.0)
        F0_spec_tint = c_tint * schlick_R0_eta(eta_it)
        F_schlick += \
                (1.0 - metallic) * spec_tint * \
                calc_schlick(F0_spec_tint, cos_theta_i, eta)

    # Front side fresnel
    F_front = (1.0 - metallic) * (1.0 - spec_tint) * F_dielectric + F_schlick
    # For back side there is no tint or metallic, just true dielectric
    # fresnel.
    return dr.select(front_side, F_front, bsdf * F_dielectric)


# ----------------------- Visualization -----------------------

def ps_visualize_textures(scene: mi.Scene, init_ps = True):
    '''
    Interactively visualize the vertex BSDF attributes using Polyscope.
    '''
    meshes = [shape for shape in scene.shapes() if shape.is_mesh()]

    if init_ps:
        ps.init()
        # ps.reset_camera_to_home_view()
        ps.look_at([0,0,-4], [0,0,0])

    bsdf_keys = [
        'vertex_bsdf_base_color',
        'vertex_bsdf_roughness',
        'vertex_bsdf_metallic',
        'vertex_bsdf_anisotropic',
        'vertex_bsdf_spec_tint',
        ]
    for idx, mesh in enumerate(meshes):
        keys = [key for key in bsdf_keys if mesh.has_attribute(key)]
        plot_mesh_attributes(mesh, f"mesh{idx}", keys, is_color=True)

    if init_ps:
        ps.show()

def render_base_color(scene: mi.Scene, img_res = None, write_img = True, filename: str = "optimized"):
    '''
    Render the vertex BSDF base colors to an OpenEXR image.
    '''
    if img_res is None:
        film_size = scene.sensors()[0].film().size()
        img_res = (film_size[0], film_size[1])
    us, vs = dr.meshgrid(dr.linspace(Float, 0.0, 1.0, img_res[0]), dr.linspace(Float, 0.0, 1.0, img_res[1]))
    sensor = scene.sensors()[0]
    rays, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(us, vs), dr.zeros(mi.Point2f))
    si = scene.ray_intersect(rays)
    mesh = si.shape
    image = dr.select(si.is_valid(), 
                    mesh.eval_attribute_3("vertex_bsdf_base_color", si),
                    mi.Color3f(0.0)
                    )

    image_out = mi.TensorXf(dr.ravel(image), shape=(img_res[0], img_res[1], 3))
    
    if write_img:
        mi.Bitmap(image_out).write(filename + ".exr")

    return image_out

def render_attributes(scene: mi.Scene, img_res = None, write_img = True, filename: str = "optimized"):
    '''
    Render all the vertex BSDF attributes to an OpenEXR image.
    '''
    if img_res is None:
        film_size = scene.sensors()[0].film().size()
        img_res = (film_size[0], film_size[1])
    us, vs = dr.meshgrid(dr.linspace(Float, 0.0, 1.0, img_res[0]), dr.linspace(Float, 0.0, 1.0, img_res[1]))
    sensor = scene.sensors()[0]
    rays, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(us, vs), dr.zeros(mi.Point2f))
    si = scene.ray_intersect(rays)
    mesh = si.shape
    mask = si.is_valid()

    # Handle the base color first
    image_rgb = dr.select(mask, 
                    mesh.eval_attribute_3("vertex_bsdf_base_color", si),
                    mi.Color3f(0.0))
    images = [image_rgb]
    channel_counts = [3]
    total_channels = 3

    # Handle the scalar attributes
    keys = ['base_color', 'roughness', 'metallic', 'anisotropic', 'spec_tint']
    for key in keys[1:]:
        attrib = f"vertex_bsdf_{key}"
        if dr.any(mesh.has_attribute(attrib)):
            image = dr.select(si.is_valid(), 
                            mesh.eval_attribute_1(attrib, si),
                            0.0)
        else:
            image = dr.full(Float, 0.0, img_res[0] * img_res[1])
        images.append(image)
        channel_counts.append(1)
        total_channels += 1

    image_out = dr.full(mi.TensorXf, 0.0, (*img_res, total_channels))

    curr_channel = 0
    for image, channel_count, key in zip(images, channel_counts, keys):
        image_out[:, :, curr_channel: curr_channel + channel_count] = dr.ravel(image)
        curr_channel += channel_count
    
    channel_names = [keys[0] + color for color in ['.R', '.G', '.B']] + keys[1:]
    if write_img:
        bitmap = mi.Bitmap(image_out, 
                           pixel_format = mi.Bitmap.PixelFormat.MultiChannel, 
                           channel_names = channel_names)
        bitmap.write(filename + f"_multichannel.exr")
    return image_out


def render_attributes_gt(scene: mi.Scene, img_res = None, write_img = True, filename: str = "optimized"):
    '''
    Render the BSDF attributes from the ground truth scene to an OpenEXR image.
    '''
    if img_res is None:
        film_size = scene.sensors()[0].film().size()
        img_res = (film_size[0], film_size[1])
    us, vs = dr.meshgrid(dr.linspace(Float, 0.0, 1.0, img_res[0]), dr.linspace(Float, 0.0, 1.0, img_res[1]))
    sensor = scene.sensors()[0]
    rays, _ = sensor.sample_ray(0.0, 0.0, mi.Point2f(us, vs), dr.zeros(mi.Point2f))
    si = scene.ray_intersect(rays)
    mesh = si.shape
    bsdf = mesh.bsdf()
    mask = si.is_valid()

    albedo_key = "base_color" if dr.any(bsdf.has_attribute("base_color")) else "reflectance"
    # Handle the base color first
    image_rgb = dr.select(mask, 
                    bsdf.eval_diffuse_reflectance(si),
                    mi.Color3f(0.0))
    images = [image_rgb]
    channel_counts = [3]
    total_channels = 3

    # Handle the scalar attributes
    keys = [albedo_key, 'roughness', 'metallic', 'anisotropic', 'spec_tint']
    for key in keys[1:]:
        image = dr.select(si.is_valid(), 
                        bsdf.eval_attribute_1(key, si),
                        0.0)
        images.append(image)
        channel_counts.append(1)
        total_channels += 1

    image_out = dr.full(mi.TensorXf, 0.0, (*img_res, total_channels))

    curr_channel = 0
    for image, channel_count, key in zip(images, channel_counts, keys):
        image_out[:, :, curr_channel: curr_channel + channel_count] = dr.ravel(image)
        curr_channel += channel_count
    
    channel_names = [keys[0] + color for color in ['.R', '.G', '.B']] + keys[1:]
    if write_img:
        bitmap = mi.Bitmap(image_out, 
                           pixel_format = mi.Bitmap.PixelFormat.MultiChannel, 
                           channel_names = channel_names)
        bitmap.write(filename + f"_multichannel.exr")
    return image_out
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
This script implements a simple optimization pipeline that refines a given set of
volumetric primitives (e.g. 3DG) given a set of reference images.
'''
import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import os, time, argparse, json
from os.path import join

import numpy as np

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

import volprim
from volprim.integrators.common import *

# Script arguments -------------------------------------------------------------

parser = argparse.ArgumentParser(description='Refine 3DG dataset')

# Required arguments
parser.add_argument('--output',     type=str, required=True, help='Path to the result output folder')
parser.add_argument('--ply',        type=str, required=True, help='Path to PLY 3DG file')
parser.add_argument('--images',     type=str, required=True, help='Path to the reference images')
parser.add_argument('--cameras',    type=str, required=True, help='Path to json camera file')

# Cameras and reference images parameters
parser.add_argument('--cam_count',      type=int,   default=8,      help='Number of cameras to use in the optimization process')
parser.add_argument('--cam_scale',      type=float, default=1.0,    help='Scaling factor for the camera resolution')
parser.add_argument('--ref_images_ext', type=str,   default='jpg',  help='Extension for the reference images')

# Integrator parameters
parser.add_argument('--ref_spp',   type=int, default=32,         help='Number of samples per pixel used for rendering the reference images')
parser.add_argument('--opt_spp',   type=int, default=1,          help='Number of samples per pixel used for rendering during the optimization')
parser.add_argument('--grad_spp',  type=int, default=1,          help='Number of samples per pixel used for gradient computation during the optimization')
parser.add_argument('--max_depth', type=int, default=128,        help='Maximum path depth for integrator')
parser.add_argument('--rr_depth',  type=int, default=256,        help='Depth at which Russian Roulette starts')
parser.add_argument('--kernel',    type=str, default='gaussian', help='Kernel type')
parser.add_argument('--white_background', action='store_true',   help='Whether to render a white background behind the 3DGs')

# Optimization parameters
parser.add_argument('--iterations',        type=int,   default=64,     help='Number of iterations for the gradient descent optimization')
parser.add_argument('--write_image_every', type=int,   default=4,      help='Frequency as which images and stats are computed')
parser.add_argument('--global_lr',         type=float, default=1.0,    help='Global learning rate')
parser.add_argument('--centers_lr',        type=float, default=0.0001, help='Learning rate multiplier for the 3DG center parameters')
parser.add_argument('--scales_lr',         type=float, default=0.0001, help='Learning rate multiplier for the 3DG scale parameters')
parser.add_argument('--quats_lr',          type=float, default=0.0001, help='Learning rate multiplier for the 3DG quat parameters')
parser.add_argument('--opacities_lr',      type=float, default=0.0001, help='Learning rate multiplier for the 3DG opacity parameters')
parser.add_argument('--sh_coeffs_lr',      type=float, default=0.002,  help='Learning rate multiplier for the 3DG SH coefficients parameters')
args = parser.parse_args()

# Prepare output folder
os.makedirs(args.output, exist_ok=True)
os.makedirs(join(args.output, 'frames'), exist_ok=True)

# Prepare scene, cameras and integrator ----------------------------------------

scene_dict = {
    'type': 'scene',
    'integrator': {
        'type': 'volprim_rf',
        'max_depth': args.max_depth,
        'rr_depth':  args.rr_depth,
        'kernel_type': args.kernel,
    },
    'primitives': {
        'type': 'ellipsoidsmesh',
        'filename': args.ply
    },
}
if args.white_background:
    scene_dict['background'] = { 'type': 'constant' }

# Add the camera dictionaries to the scene description
cam_specs = volprim.cameras.JSONCameraSpecsIO.load(args.cameras)

cam_indices = list(range(0, len(cam_specs), len(cam_specs) // args.cam_count))[:args.cam_count]
print(f'Camera indices: {cam_indices}')

# Add the camera dictionaries to the scene description
for i in cam_indices:
    scene_dict[cam_specs[i].name] = cam_specs[i].to_dict(args.cam_scale)
scene = mi.load_dict(scene_dict)
params = mi.traverse(scene)

# Create batch sensor -----------------------------------------

res = scene.sensors()[0].film().crop_size()
batch_sensor_dict = {
    'type': 'batch',
    'film': {
        'type': 'hdrfilm',
        'width': res[0] * args.cam_count, 'height': res[1],
        'filter': { 'type': 'tent' },
    }
}
for i in cam_indices:
    batch_sensor_dict[f'cam_{i:04d}'] = mi.load_dict(cam_specs[i].to_dict(args.cam_scale))
batch_sensor = mi.load_dict(batch_sensor_dict)

# Load reference images --------------------------------------------------------

ref_images = []
for i in cam_indices:
    img_name = cam_specs[i].name
    bmp = mi.Bitmap(join(args.images, f'{img_name}.{args.ref_images_ext}'))
    bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    bmp = bmp.resample(res)
    ref_images.append(mi.TensorXf(bmp))
ref_image = volprim.utils.concatenate_tensors(ref_images)
print(f"Writing image to '{join(args.output, 'reference.exr')}'...")
mi.util.write_bitmap(join(args.output, 'reference.exr'), ref_image)

# Render initial scene -----------------------------------------------------------

with volprim.utils.time_operation('Render initial images'):
    init_image = mi.render(scene, sensor=batch_sensor, spp=args.ref_spp)
print(f"Writing image to '{join(args.output, 'initial.exr')}'...")
mi.util.write_bitmap(join(args.output, 'initial.exr'), init_image)

# Setup optimization -----------------------------------------------------------

key_data      = 'primitives.data'
key_opacities = 'primitives.opacities'
key_sh_coeffs = 'primitives.sh_coeffs'

opt = volprim.optimizers.BoundedAdam()

ellipsoids = Ellipsoid.unravel(params[key_data])
opt['centers'] = ellipsoids.center
opt['scales']  = ellipsoids.scale
opt['quats']   = mi.Vector4f(ellipsoids.quat)
opt['opacities'] = params[key_opacities]
opt['sh_coeffs'] = params[key_sh_coeffs]

opt.set_learning_rate({
    'centers':   args.global_lr * args.centers_lr,
    'scales':    args.global_lr * args.scales_lr,
    'quats':     args.global_lr * args.quats_lr,
    'opacities': args.global_lr * args.opacities_lr,
    'sh_coeffs': args.global_lr * args.sh_coeffs_lr,
})

opt.set_bounds('scales',    lower=1e-6)
opt.set_bounds('opacities', lower=1e-6, upper=1.0-1e-6)

def update_params(opt):
    params[key_data] = Ellipsoid.ravel(opt['centers'], opt['scales'], mi.Quaternion4f(opt['quats']))
    params[key_opacities] = opt['opacities']
    params[key_sh_coeffs] = opt['sh_coeffs']
    params.update()

update_params(opt)

# Optimize! --------------------------------------------------------------------

loss_list = []
psnr_list = []

print(f'Run optimization:')

for it in range(args.iterations):
    image = mi.render(scene, params, sensor=batch_sensor, spp=args.opt_spp, spp_grad=args.grad_spp, seed=it)

    dr.eval(image)

    loss = volprim.optimizers.l1(ref_image, image)
    psnr = volprim.optimizers.psnr(ref_image, image)

    dr.backward(loss)
    opt.step()
    update_params(opt)

    loss_list.append(loss.array[0])
    psnr_list.append(psnr.array[0])

    with dr.suspend_grad():
        if (it + 1) % args.write_image_every == 0:
            mi.util.write_bitmap(join(args.output, 'frames', f'image_{it:04d}.exr'), image)

    print(f'-- step {it+1} / {args.iterations} | psnr={psnr_list[-1]:.04f} | loss={loss_list[-1]:.04f}', end='\r')

print()
print('Done with optimization')

# Save results -----------------------------------------------------------------

optimized = mi.render(scene, sensor=batch_sensor, spp=args.ref_spp)
print(f"Writing image to '{join(args.output, 'optimized.exr')}'...")
mi.util.write_bitmap(join(args.output, 'optimized.exr'), optimized)

volprim.io.dict_to_asset(volprim.io.object_to_dict(scene), join(args.output, 'optimized_asset'))

print(f'PSNR: {volprim.optimizers.psnr(ref_image, optimized).array[0]}')

def plot_loss(data, label, output_file):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(data)
    fig.canvas.toolbar_visible = 'fade-in-fade-out'
    fig.canvas.footer_visible = False
    fig.canvas.header_visible = False
    ax.set_xlabel('Iteration')
    plt.ylabel(label)
    plt.title(label + ' plot')
    plt.savefig(output_file)

plot_loss(loss_list, label='Loss', output_file=join(args.output, 'loss.png'))
plot_loss(psnr_list, label='PSNR', output_file=join(args.output, 'psnr.png'))

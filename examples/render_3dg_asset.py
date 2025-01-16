# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
This script is a simple example of how to use Mitsuba to render a 3DG asset
from the original 3D Gaussian Splatting paper datasets.

Datasets can be downloaded at the following link:

https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

Example:

    python examples\render_3dg_asset.py --ply datasets\truck\point_cloud\iteration_30000\point_cloud.ply --cameras datasets\truck\cameras.json
'''

import argparse
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

import volprim

# Script arguments -------------------------------------------------------------

parser = argparse.ArgumentParser(description='Render 3DG asset')

# Required arguments
parser.add_argument('--ply',     type=str, required=True, help='Path to PLY 3DG file')
parser.add_argument('--cameras', type=str, required=True, help='Path to json camera file')
parser.add_argument('--output',  type=str, default='output.exr', help='Path to the output image')

# Cameras and reference images parameters
parser.add_argument('--cam_index',  type=int,   default=0,      help='Index of the camera to render')
parser.add_argument('--cam_scale',  type=float, default=1.0,    help='Scaling factor for the camera resolution')

# Integrator parameters
parser.add_argument('--spp',       type=int, default=2,          help='Number of samples per pixel used for rendering the images')
parser.add_argument('--max_depth', type=int, default=128,        help='Maximum path depth for integrator')
parser.add_argument('--rr_depth',  type=int, default=128,        help='Depth at which Russian Roulette starts')
parser.add_argument('--kernel',    type=str, default='gaussian', help='Kernel type')
parser.add_argument('--white_background', action='store_true',   help='Whether to render a white background behind the 3DGs')

args = parser.parse_args()

# Prepare scene, cameras and integrator
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
for spec in cam_specs:
    scene_dict[spec.name] = spec.to_dict(args.cam_scale)

# Load the scene
scene = mi.load_dict(scene_dict)

# Render the scene
with volprim.benchmark.single_run('Rendering'):
    img = mi.render(scene, sensor=args.cam_index, spp=args.spp)

# Write image to file on disk
print(f'Writing rendered image to {args.output}')
mi.util.write_bitmap(args.output, img)

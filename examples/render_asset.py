# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is a simple example of how to use Mitsuba to render a Volumetric Primitives asset.
"""
import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import argparse

# Parse script arguments
parser = argparse.ArgumentParser(description='Render a Python asset')
parser.add_argument('asset',       type=str,                  help='Path to the Python asset')
parser.add_argument('--cam_index', type=int,   default=0,     help='Index of the cameras to use for rendering')
parser.add_argument('--cam_scale', type=float, default=1.0,   help='Scaling factor for the camera resolution')
parser.add_argument('--spp',       type=int,   default=4,     help='Number of samples per pixel used for rendering')
parser.add_argument('--variant',   type=str,   default='',    help='Mitsuba variant to use for rendering')
parser.add_argument('--output',    type=str,   default='output.exr', help='Path to the output image')
args = parser.parse_args()

import mitsuba as mi
mi.set_variant(args.variant, 'cuda_ad_rgb')
import volprim

# Assemble the scene dictionary from the Python asset
scene_dict = volprim.io.asset_to_dict(args.asset)

# Scale down the film resolution of the cameras defined by the asset
scene_dict = volprim.io.scale_films(scene_dict, args.cam_scale)

# Load the scene with Mitsuba
scene = mi.load_dict(scene_dict)

# Render the scene
with volprim.benchmark.single_run('Rendering'):
    img = mi.render(scene, sensor=args.cam_index, spp=args.spp)

# Write image to file on disk
print(f'Writing rendered image to {args.output}')
mi.util.write_bitmap(args.output, img)

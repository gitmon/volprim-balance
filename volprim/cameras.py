# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
This module provides input/output routines for handling camera specifications
in various formats. The module supports loading and writing camera data
from JSON, KRT, and Colmap formats, and provides utility functions for
converting between different camera parameter representations

- JSONCameraSpecsIO:   Handles loading and writing camera data from JSON files.
- KRTCameraSpecsIO:    Handles loading and writing camera data from KRT JSON files.
- ColmapCameraSpecsIO: Handles loading camera data from Colmap model files.

Example:

    cam_specs = JSONCameraSpecsIO.load('cameras.json')
    mitsuba_camera_dicts = [spec.to_dict() for spec in cam_specs]
'''

from __future__ import annotations # Delayed parsing of type annotations
from typing import List

from . import colmap_loader

import os
import numpy as np
import drjit as dr
import mitsuba as mi

# Notes regarding conventions:
#
# Sensors in Mitsuba 3 are right-handed. Any number of rotations and translations
# can be applied to them without changing this property. By default, they are located
# at the origin and oriented in such a way that in the rendered image,
# points left, points upwards, and points along the viewing direction.
# Left-handed sensors are also supported. To switch the handedness, flip any one of the
# axes, e.g. by passing a scale transform like <scale x="-1"/> to the sensor’s to_world parameter.

def fov2focal(fov: float, width: int):
    '''
    Compute the focal length (pixel) for a given sensor resolution and FOV (degree)
    '''
    return (width / 2.0) / dr.tan(dr.deg2rad(fov) * 0.5)

def focal2fov(focal_length: float, width: int):
    '''
    Compute the FOV (degrees) for a given sensor resolution and focal length
    '''
    return 2.0 * dr.rad2deg(dr.atan2(0.5 * width, focal_length))

class CameraSpecs:
    '''
    Camera information data structure.
    '''
    def __init__(self,
                 name: str,
                 width: int,
                 height: int,
                 to_world: mi.ScalarTransform4f,
                 # Field of view (degrees)
                 fov: float=None,
                 # Focal length (pixels)
                 focal_length: float=None,
                 # Near and far clips (meters)
                 near_clip: float=0.1,
                 far_clip: float=10000.0,
                 # Principal point offsets ([-1;1])
                 cx: float=0.0,
                 cy: float=0.0,
                 # Radial distortion coefficients
                 k1: float=0.0,
                 k2: float=0.0,
                 k3: float=0.0,
                 k4: float=0.0,
                 k5: float=0.0,
                 k6: float=0.0,
                 # Tangential distortion coefficients
                 p1: float=0.0,
                 p2: float=0.0):
        self.name = name
        self.width, self.height = width, height
        self.to_world = mi.ScalarTransform4f(to_world)
        self.fov, self.focal_length = fov, focal_length
        self.near_clip, self.far_clip = near_clip, far_clip
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = k1, k2, k3, k4, k5, k6
        self.p1, self.p2 = p1, p2
        if self.fov is None:
            self.fov = focal2fov(self.focal_length, self.width)
        elif self.focal_length is None:
            self.focal_length = fov2focal(self.fov, self.width)
        else:
            raise Exception('CameraSpecs: either FOV or focal length should be set!')

    def viewmat(self) -> np.array:
        '''
        Return the view matrix (i.e. world-to-cam transformation transform)
        using the convention of GSplat.
        '''
        return np.array(self.to_world.scale([-1, -1, 1]).inverse().matrix)

    def K(self) -> np.array:
        '''
        Return the camera intrinsics matrix
        '''
        return np.array([
            [self.focal_length, 0.0, self.width / 2.0],
            [0.0, self.focal_length, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ])

    def to_dict(self,
                resolution_factor: float=1.0,
                pixel_format: str='rgb',
                pixel_filter: str='tent') -> dict:
        '''
        Generate corresponding Mitsuba dictionary
        '''
        return {
            'type': 'perspective',
            'principal_point_offset_x': self.cx,
            'principal_point_offset_y': self.cy,
            'fov_axis': 'x',
            'fov': self.fov,
            'to_world': self.to_world,
            'near_clip': self.near_clip,
            'far_clip': self.far_clip,
            'film': {
                'type': 'hdrfilm',
                'rfilter': { 'type': pixel_filter },
                'pixel_format': pixel_format,
                'width':  int(self.width * resolution_factor),
                'height': int(self.height * resolution_factor),
            }
        }

    @staticmethod
    def from_dict(d: dict, name: str=''):
        return CameraSpecs(
            name = name,
            to_world = d['to_world'],
            fov = d['fov'],
            width = d['film']['width'],
            height = d['film']['height'],
            cx = d.get('principal_point_offset_x', 0.0),
            cy = d.get('principal_point_offset_y', 0.0),
            near_clip = d.get('near_clip', 0.1),
            far_clip = d.get('far_clip', 10000.0)
        )

    def __repr__(self):
        return f"CameraSpecs[\n" + '\n'.join([f"  {k}: {v}" for k, v in self.__dict__.items()]) + "]"

# ------------------------------------------------------------------------------

class CameraSpecsIO:
    @staticmethod
    def load(filename: str) -> List[CameraSpecs]:
        raise Exception('Loader not implemented')

    @staticmethod
    def write(specs: List[CameraSpecs], filename: str):
        raise Exception('Loader not implemented')

# ------------------------------------------------------------------------------

class JSONCameraSpecsIO(CameraSpecsIO):
    '''
    Load / write sensor dictionaries from json file (e.g. 3DG datasets)
    '''
    @staticmethod
    def load(filename: str) -> List[CameraSpecs]:
        import json
        with open(filename) as f:
            sensors = json.load(f)

        specs = []
        for sensor in sensors:
            to_world = np.eye(4)
            to_world[:3, :3] = np.array(sensor['rotation']).transpose(0, 1)
            to_world[:3, 3]  = np.array(sensor['position'])

            to_world = mi.ScalarTransform4f(to_world).scale([-1, -1, 1])

            specs.append(CameraSpecs(
                name=sensor['img_name'],
                width = sensor['width'],
                height = sensor['height'],
                focal_length = sensor['fx'],
                to_world = to_world,
                near_clip = 0.01 * 10,
                far_clip = 100.0,
            ))

        return specs

    @staticmethod
    def write(specs: List[CameraSpecs], filename: str):
        sensors = []
        for i, cam in enumerate(specs):
            sensor = {}
            to_world = cam.to_world @ mi.ScalarTransform4f().scale([-1, -1, 1])
            sensor['rotation'] = np.array(to_world.matrix)[:3, :3].transpose(0, 1).tolist()
            sensor['position'] = np.array(to_world.matrix)[:3, 3].tolist()
            sensor['fx'] = cam.focal_length
            sensor['fy'] = cam.focal_length
            sensor['width']  = cam.width
            sensor['height'] = cam.height
            sensor['id'] = i
            sensor['img_name'] = cam.name
            sensors.append(sensor)

        import json
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(json.dumps(sensors, ensure_ascii=False))

# ------------------------------------------------------------------------------

class KRTCameraSpecsIO(CameraSpecsIO):
    '''
    Load / write camera information from a KRT json file (e.g. Digital Human Studio)
    '''
    @staticmethod
    def load(filename: str) -> List[CameraSpecs]:
        import json
        with open(filename) as f:
            sensors = json.load(f)['KRT']

        infos = []
        for sensor in sensors:
            # TODO
            if sensor['distortionModel'] != 'RadialAndTangential':
                continue

            # TODO
            if sensor['projectionModel'] != 'Pinhole':
                continue

            K = np.array(sensor['K'])
            RT = np.array(sensor['T'])
            k1, k2, k3, k4 = list(sensor['distortion'][0])

            px, py = K[2, 1], K[2, 1]
            width, height = 2 * px, 2 * py # TODO this assumes principal points are in the center

            infos.append(CameraSpecs(
                name=sensor['cameraId'],
                width = width,
                height = height,
                to_world = RT,
                focal_length = K[0, 0],
                cx = 0,
                cy = 0,
                k1 = k1,
                k2 = k2,
                k3 = k3,
                k4 = k4,
            ))

        return infos

# ------------------------------------------------------------------------------

class ColmapCameraSpecsIO(CameraSpecsIO):
    '''
    Load sensor info from colmap model
    '''
    @staticmethod
    def load(filename: str) -> List[CameraSpecs]:
        try:
            cameras_extrinsic_file = os.path.join(filename, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(filename, "sparse/0", "cameras.bin")
            cam_extrinsics = colmap_loader.read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = colmap_loader.read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(filename, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(filename, "sparse/0", "cameras.txt")
            cam_extrinsics = colmap_loader.read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = colmap_loader.read_intrinsics_text(cameras_intrinsic_file)

        infos = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(colmap_loader.qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            k1 = k2 = k3 = k4 = k5 = k6 = p1 = p2 = 0.0

            # https://github.com/colmap/colmap/blob/a7a0db7d2ae0ca9e6536bc10c96a9b6c3a2578a9/src/colmap/sensor/models.h#L251
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model=="SIMPLE_RADIAL":
                focal_length_x = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
                k1 = intr.params[3]
            elif intr.model=="RADIAL":
                focal_length_x = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
                k1 = intr.params[3]
                k2 = intr.params[4]
            elif intr.model=="OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                k1 = intr.params[4]
                k2 = intr.params[5]
                p1 = intr.params[6]
                p2 = intr.params[7]
            elif intr.model=="OPENCV_FISHEYE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                k1 = intr.params[4]
                k2 = intr.params[5]
                k3 = intr.params[6]
                k4 = intr.params[7]
            elif intr.model=="FULL_OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
                k1 = intr.params[4]
                k2 = intr.params[5]
                p1 = intr.params[6]
                p2 = intr.params[7]
                k3 = intr.params[8]
                k4 = intr.params[9]
                k5 = intr.params[10]
                k6 = intr.params[11]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            to_cam = np.eye(4)
            to_cam[:3, :3] = R * [-1,-1, 1]
            to_cam[3, :3]  = T * [-1,-1, 1]

            to_world = mi.ScalarTransform4f(np.linalg.inv(to_cam).transpose())

            infos.append(CameraSpecs(
                name=extr.name.replace(".", "_"),
                width = width,
                height = height,
                to_world = to_world,
                focal_length = focal_length_x,
                cx = width / 2.0 - cx,
                cy = height / 2.0 - cy,
                k1 = k1,
                k2 = k2,
                k3 = k3,
                k4 = k4,
                k5 = k5,
                k6 = k6,
                p1 = p1,
                p2 = p2,
            ))

        return infos

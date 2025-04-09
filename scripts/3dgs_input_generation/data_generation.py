import os
import sys
sys.path.insert(0, '/home/jonathan/Documents/mi3-gs/build/python')

import numpy as np

import drjit as dr
from drjit.auto.ad import Float, UInt
import mitsuba as mi
import open3d as o3d

# HOME_DIR = "/home/jonathan/Documents/gaussian-splatting/datasets/mitsuba"
HOME_DIR = "/home/jonathan/Documents/volprim-balance/3dgs_input"

# ----------------- Point cloud generation ------------------

def generate_point_cloud(scene: mi.Scene, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a point cloud on the surfaces of the scene.
    """
    shapes = scene.shapes_dr()
    sampler = mi.load_dict({'type':'independent'})
    sampler.seed(0, wavefront_size = num_points)
    shape_distr = mi.DiscreteDistribution(dr.ones(Float, dr.width(shapes))) # replace with equi-area?

    sampled_idxs = shape_distr.sample(sampler.next_1d())
    sampled_shape = dr.gather(mi.ShapePtr, shapes, sampled_idxs)
    surface_samples = sampled_shape.sample_position(0.0, sampler.next_2d())

    si = mi.SurfaceInteraction3f(surface_samples, mi.Color0f())
    surface_colors = sampled_shape.bsdf().eval_diffuse_reflectance(si)

    point_positions = si.p.numpy().T
    point_colors = surface_colors.numpy().T
    point_normals = si.n.numpy().T
    return {
        "positions": point_positions, 
        "colors": point_colors, 
        "normals": point_normals,
        }


def write_point_cloud(scene: mi.Scene, output_path: str, num_points: int, filename: str = "points3d.ply") -> None:
    """
    Synthesize the point cloud and write it to a .PLY file. GS/NeRF requires that the point cloud
    contain three attributes: the vertex positions, colors, and surface normals.
    """
    points = generate_point_cloud(scene, num_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector( points["positions"])
    pcd.colors = o3d.utility.Vector3dVector( points["colors"])
    pcd.normals = o3d.utility.Vector3dVector(points["normals"])
    fp = os.path.join(output_path, filename)
    o3d.io.write_point_cloud(fp, pcd)

# ----------------- Camera pose generation ------------------

from gpytoolbox import icosphere, per_face_normals
from typing import NamedTuple, Generator, Iterable

class CameraPose(NamedTuple):
    """
    Camera pose container. One container can store multiple (vectorized) poses.
    """
    origin: np.ndarray
    target: np.ndarray
    up: np.ndarray

class CameraParameters(NamedTuple):
    """
    Container for camera intrinsic parameters. 
    """
    width: int
    height: int
    spp: int
    fov: float

def make_spherical_cameras(
        center: np.ndarray = np.zeros(3), 
        radius: float = 1.0, 
        outward: bool = False, 
        density: int = 0) -> CameraPose:
    V, F = icosphere(density)
    V = radius * V + center[None, :]
    N = per_face_normals(V, F)
    origins = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0
    targets = origins + (N if outward else -N)
    ups = np.zeros_like(origins); ups[:,1] = 1.0
    return CameraPose(origins, targets, ups)

def make_hemispherical_cameras(
        center: np.ndarray = np.zeros(3), 
        radius: float = 1.0, 
        outward: bool = False, 
        tophalf: bool = False,
        density: int = 0) -> CameraPose:
    """
    Create a set of camera poses evenly distributed on an icosphere. The "density" parameter controls the 
    icosphere's subdivision level, which in turn determines the total number of camera poses generated.
    """
    V, F = icosphere(density)
    V = radius * V + center[None, :]
    N = per_face_normals(V, F)
    origins = (V[F[:,0]] + V[F[:,1]] + V[F[:,2]]) / 3.0
    targets = origins + (N if outward else -N)
    ups = np.zeros_like(origins); ups[:,1] = 1.0

    mask = (origins[:,1] > center[1]) if tophalf else np.ones(origins.shape[0]).astype(bool)
    origins = origins[mask]
    targets = targets[mask]
    ups = ups[mask]
    return CameraPose(origins, targets, ups)

def concatenate_cameras(pose_sets: Iterable[CameraPose]):
    origins = [pose_set.origin for pose_set in pose_sets]
    targets = [pose_set.target for pose_set in pose_sets]
    ups = [pose_set.up for pose_set in pose_sets]
    return CameraPose(
        np.vstack(origins),
        np.vstack(targets),
        np.vstack(ups),
        )


def create_camera(camera_params: CameraParameters, poses: CameraPose) -> Iterable[dict]:
    """
    Generate Mitsuba sensor dicts for a perspective camera with intrinsic parameters given by `camera_params`,
    and a set of input camera poses.
    """
    dicts = []
    for origin, target, up in zip(poses.origin, poses.target, poses.up):
        sensor_dict = {
            'type': 'perspective',
            'fov': camera_params.fov,
            'to_world': mi.ScalarTransform4f().look_at(
                origin=tuple(origin),
                target=tuple(target),
                up=tuple(up)
            ),
            'film_id': {
                'type': 'hdrfilm',
                'width':  camera_params.width,
                'height': camera_params.height,
                # Box reconstruction filter for denoising
                'filter': { 'type': 'box' }
            },
            'sampler_id': {
                'type': 'independent',
                'sample_count': camera_params.spp,
            }
        }
        dicts.append(sensor_dict)
    return dicts

# def create_cameras_batch(camera_params: CameraParameters, poses: CameraPose) -> dict:
#     num_cameras = poses.origin.shape[0]
#     sensor_dict = {
#             'type': 'batch',
#             'film_id': {
#                 'type': 'hdrfilm',
#                 'width':  camera_params.width * num_cameras,
#                 'height': camera_params.height,
#                 # Use a Gaussian reconstruction filter
#                 'filter': { 'type': 'gaussian' }
#             },
#             'sampler_id': {
#                 'type': 'independent',
#                 'sample_count': camera_params.spp,
#             }
#         }
    
#     for camera_id, (origin, target, up) in enumerate(zip(poses.origin, poses.target, poses.up)):
#         sensor_dict[f'sensor{camera_id}'] = {
#             'type': 'perspective',
#             'fov': camera_params.fov,
#             'to_world': mi.ScalarTransform4f().look_at(
#                 origin=tuple(origin),
#                 target=tuple(target),
#                 up=tuple(up)
#             )
#         }
#     return sensor_dict

# ----------------- Camera rendering and json output ------------------

import json
from enum import Enum
from tqdm import tqdm

class DataSplit(Enum):
    Train = 0
    Test = 1

def render_views(scene: mi.Scene, output_path: str, camera_params: CameraParameters, camera_poses: CameraPose, split: DataSplit = DataSplit.Train, denoise: bool = True) -> None:
    """
    Render the views from the set of input camera poses and write them to disk.
    """
    split_name = "train" if split == DataSplit.Train else "test"

    if not(denoise):
        # Without denoising
        for camera_id, sensor_dict in enumerate(tqdm(create_camera(camera_params, camera_poses))):
            sensor = mi.load_dict(sensor_dict)
            image = mi.render(scene, sensor=sensor)
            fp = os.path.join(output_path, split_name, f"sensor_{camera_id}.png")
            mi.util.write_bitmap(fp, image, write_async=True)

    else:
        # With denoising
        integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'albedo:albedo,normals:sh_normal',
            'integrator': {
                'type': 'path',
                'max_depth': 17,
                }
            })
        
        # Denoise the rendered image
        img_size = mi.ScalarVector2u(camera_params.width, camera_params.height)
        denoiser = mi.OptixDenoiser(input_size=img_size, albedo=True, normals=True, temporal=False)

        for camera_id, sensor_dict in enumerate(tqdm(create_camera(camera_params, camera_poses))):
            sensor = mi.load_dict(sensor_dict)
            to_sensor = sensor.world_transform().inverse()
            mi.render(scene, sensor=sensor, integrator=integrator)
            image = sensor.film().bitmap()
            denoised = denoiser(image, albedo_ch="albedo", normals_ch="normals", to_sensor=to_sensor)

            fp = os.path.join(output_path, split_name, f"sensor_{camera_id}.png")
            mi.util.write_bitmap(fp, denoised, write_async=True)

def write_poses_to_json(output_path: str, camera_params: CameraParameters, poses: CameraPose, split: DataSplit = DataSplit.Train) -> None:
    """
    Write the input camera poses to disk as a .json file.
    """
    split_name = "train" if split == DataSplit.Train else "test"
    # Specify intrinsic parameters
    json_data = {
        # # Horizontal and vertical fov
        "camera_angle_x": dr.deg2rad(camera_params.fov),
        # "camera_angle_y": None,
        # # Image size in pixels
        'w': camera_params.width,
        'h': camera_params.height,
        # Principal point position
        'cx': camera_params.width / 2,
        'cy': camera_params.height / 2,
        # # Radial distortion coefficients (barrel/pincushion/etc.)
        # 'k1': None,
        # 'k2': None,
        # # Tangential distortion coefficients (incline)
        # 'p1': None,
        # 'p2': None,
    }

    # Specify camera world poses
    frames = []
    for camera_id, (origin, target, up) in enumerate(zip(poses.origin, poses.target, poses.up)):
        frame = {
            "file_path": os.path.join(split_name, f"sensor_{camera_id}"),
            # "rotation": 0.0,
        }
        to_world_matrix = mi.ScalarTransform4f().look_at(tuple(origin), tuple(target), tuple(up))
        to_world_matrix = to_world_matrix.matrix.numpy()
        # Appropriate coordinate system transform can be found in the Instant-NGP repo:
        # https://github.com/NVlabs/instant-ngp/blob/5595c47639ab495bad66f4d661ea2720d30befa6/include/neural-graphics-primitives/nerf_loader.h#L108
        to_world_matrix = to_world_matrix.astype(np.double) @ np.diag([-1, 1, -1, 1])
        
        frame["transform_matrix"] = [
            list(to_world_matrix[0]),
            list(to_world_matrix[1]),
            list(to_world_matrix[2]),
            list(to_world_matrix[3]),
        ]
        frames.append(frame)

    json_data["frames"] = frames
    filepath = os.path.join(output_path, f"transforms_{split_name}.json")
    with open(filepath, 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


# ----------------- Main data generation ------------------

import polyscope as ps

class DataGenerator:
    def __init__(self, scene: mi.Scene, dataset_name: str, camera_params: CameraParameters, camera_poses: CameraPose, pointcloud_size: int = 1 << 20, use_denoiser: bool = False):
        self.scene = scene
        self.output_path = os.path.join(HOME_DIR, dataset_name)
        self.camera_params = camera_params
        self.camera_poses = camera_poses
        self.pointcloud_size = pointcloud_size
        self.use_denoiser = use_denoiser

    def run(self):
        try:
            os.makedirs(os.path.join(self.output_path, "test"))
        except FileExistsError:
            # directory already exists
            pass

        try:
            os.makedirs(os.path.join(self.output_path, "train"))
        except FileExistsError:
            # directory already exists
            pass

        poses_train = self.camera_poses
        poses_test = CameraPose(
            poses_train.origin[::4],
            poses_train.target[::4],
            poses_train.up[::4])

        # Generate train data
        print("Starting render job for training data ...")
        render_views(self.scene, self.output_path, self.camera_params, poses_train, split=DataSplit.Train, denoise=self.use_denoiser)
        write_poses_to_json(     self.output_path, self.camera_params, poses_train, split=DataSplit.Train)
        print("Training data render complete.")

        # Generate test data
        print("Starting render job for test data ...")
        render_views(self.scene, self.output_path, self.camera_params, poses_test, split=DataSplit.Test, denoise=self.use_denoiser)
        write_poses_to_json(     self.output_path, self.camera_params, poses_test, split=DataSplit.Test)
        print("Test data render complete.")

        # Generate point cloud
        write_point_cloud(self.scene, self.output_path, self.pointcloud_size)

    def visualize(self):
        ps.init()

        pcd = generate_point_cloud(self.scene, self.pointcloud_size)
        plot = ps.register_point_cloud("Points", pcd["positions"])
        plot.add_color_quantity("Color", pcd["colors"], enabled=True)
        plot.add_vector_quantity("Normal", pcd["normals"])

        intrinsics = ps.CameraIntrinsics(
            fov_horizontal_deg=self.camera_params.fov, 
            aspect=float(self.camera_params.width) / self.camera_params.height)
        poses = self.camera_poses
        for camera_id, (origin, target, up) in enumerate(zip(poses.origin, poses.target, poses.up)):
            extrinsics = ps.CameraExtrinsics(root=origin, look_dir=target-origin, up_dir=up)
            params = ps.CameraParameters(intrinsics, extrinsics)
            ps.register_camera_view(f"cam{camera_id}", params)

        ps.show()




def render_views_HDR(scene: mi.Scene, output_path: str, camera_params: CameraParameters, camera_poses: CameraPose, denoise: bool = True) -> None:
    """
    Render the views from the set of input camera poses and write them to disk.
    """
    # split_name = "train" if split == DataSplit.Train else "test"

    if denoise:
        integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'albedo:albedo,normals:sh_normal',
            'integrator': {
                'type': 'path',
                'max_depth': 17,
                }
            })
        img_size = mi.ScalarVector2u(camera_params.width, camera_params.height)
        denoiser = mi.OptixDenoiser(input_size=img_size, albedo=True, normals=True, temporal=False)

    for camera_id, sensor_dict in enumerate(tqdm(create_camera(camera_params, camera_poses))):
        sensor = mi.load_dict(sensor_dict)
        if denoise:
            mi.render(scene, sensor=sensor, integrator=integrator)
            noisy = sensor.film().bitmap()
            to_sensor = sensor.world_transform().inverse()
            denoised = denoiser(noisy, albedo_ch="albedo", normals_ch="normals", to_sensor=to_sensor)
            image = denoised
        else:
            image = mi.render(scene, sensor=sensor)

        # write HDR image
        fp = os.path.join(output_path, "exr", f"{camera_id}.exr")
        mi.util.write_bitmap(fp, image, write_async=True)

        # write LDR images
        exposures = [0.2, 0.4, 0.6, 0.8, 1.0]
        for exp_id, scale_factor in enumerate(exposures):
            fp = os.path.join(output_path, "images", f"{camera_id}_{exp_id}.png")
            _image = mi.Bitmap(scale_factor * mi.TensorXf(image))
            mi.util.write_bitmap(fp, _image, write_async=True)


def write_poses_to_json_HDR(output_path: str, camera_params: CameraParameters, poses: CameraPose) -> None:
    """
    Write the input camera poses to disk as a .json file.
    """
    # Specify intrinsic parameters
    json_data = {
        "camera_angle_x": dr.deg2rad(camera_params.fov),
        'w': camera_params.width,
        'h': camera_params.height,
        'cx': camera_params.width / 2,
        'cy': camera_params.height / 2,
    }

    # Specify camera world poses
    frames = []
    for camera_id, (origin, target, up) in enumerate(zip(poses.origin, poses.target, poses.up)):
        frame = {
            "file_path": f"{camera_id}",
        }
        to_world_matrix = mi.ScalarTransform4f().look_at(tuple(origin), tuple(target), tuple(up))
        to_world_matrix = to_world_matrix.matrix.numpy()
        # Appropriate coordinate system transform can be found in the Instant-NGP repo:
        # https://github.com/NVlabs/instant-ngp/blob/5595c47639ab495bad66f4d661ea2720d30befa6/include/neural-graphics-primitives/nerf_loader.h#L108
        to_world_matrix = to_world_matrix.astype(np.double) @ np.diag([-1, 1, -1, 1])
        
        frame["transform_matrix"] = [
            list(to_world_matrix[0]),
            list(to_world_matrix[1]),
            list(to_world_matrix[2]),
            list(to_world_matrix[3]),
        ]
        frames.append(frame)

    json_data["frames"] = frames
    filepath = os.path.join(output_path, f"transforms_train.json")
    with open(filepath, 'w') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


class DataGeneratorHDR(DataGenerator):
    def __init__(self, scene: mi.Scene, dataset_name: str, camera_params: CameraParameters, camera_poses: CameraPose, pointcloud_size: int = 1 << 20, use_denoiser: bool = False):
        super().__init__(scene, dataset_name, camera_params, camera_poses, pointcloud_size, use_denoiser)

    def run(self):
        try:
            os.makedirs(os.path.join(self.output_path, "exr"))
        except FileExistsError:
            # directory already exists
            pass

        try:
            os.makedirs(os.path.join(self.output_path, "images"))
        except FileExistsError:
            # directory already exists
            pass

        try:
            os.makedirs(os.path.join(self.output_path, "sparse", "0"))
        except FileExistsError:
            # directory already exists
            pass

        poses_train = self.camera_poses

        # Generate train data
        print("Starting render job for training data ...")
        render_views_HDR(self.scene, self.output_path, self.camera_params, poses_train, denoise=self.use_denoiser)
        write_poses_to_json_HDR(     self.output_path, self.camera_params, poses_train)
        print("Training data render complete.")

        # Generate point cloud
        pc_path = os.path.join(self.output_path, "sparse", "0")
        write_point_cloud(self.scene, pc_path, self.pointcloud_size, filename="points3D.ply")
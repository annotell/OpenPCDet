"""
This file is used to visualize point clouds with cuboid annotations on your LOCAL machine!
It is used to check that the projection of the fetched annotation matches the point cloud.
It connects to a remote server via SSH and downloads a random pair of point cloud and annotation files.
The point cloud is visualized using Open3D, and the cuboid annotations are overlaid on top of it.
"""

import random
import numpy as np
import open3d as o3d
import pickle
import os
from pathlib import Path
from annotell.judgement_shapes.cube_3d import Cube3D
from scipy.spatial.transform import Rotation
import json
import paramiko
from scp import SCPClient

# Set environment variables for OpenGL
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Use OSMesa for software rendering
os.environ["OSMESA_PREFIX"] = "/usr"  # Adjust this path if needed

# Get the script directory
SCRIPT_DIR = Path(__file__).parent

SSH_USERNAME = "andre"
SSH_HOST = ["158.174.46.140", "192.168.143.115"]


def check_ssh_connection(hosts, username):
    for host in hosts:
        try:
            print(f"Trying SSH to {host}...")
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            # Default policy is RejectPolicy: only connect to hosts in ~/.ssh/known_hosts
            client.connect(hostname=host, username=username, timeout=5)
            client.close()
            print(f"Connected to {host}")
            return host
        except Exception as e:
            print(f"Failed to connect to {host}: {e}")
    print("Could not connect to any SSH host. Check username, IPs, and SSH key.")
    return None


def list_remote_directories(
    host, username, remote_path="/mnt/bfd/datasets/autobaans/3dod"
):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Default policy is RejectPolicy: only connect to hosts in ~/.ssh/known_hosts
    client.connect(hostname=host, username=username)
    stdin, stdout, stderr = client.exec_command(f"ls -d {remote_path}/*/")
    dirs = [line.strip().split("/")[-2] for line in stdout.readlines()]
    client.close()

    if not dirs:
        print("No directories found.")
        return None

    print("Available projects:")
    for i, d in enumerate(dirs):
        print(f"[{i}] {d}")

    index = int(input("Choose a project by number: "))
    return dirs[index]


def count_anno_files(host, username, project_path):
    remote_annos_path = f"/mnt/bfd/datasets/autobaans/3dod/{project_path}/annos"
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Default policy is RejectPolicy: only connect to hosts in ~/.ssh/known_hosts
    client.connect(hostname=host, username=username)
    stdin, stdout, stderr = client.exec_command(f"ls {remote_annos_path} | wc -l")
    count = int(stdout.read().strip())
    client.close()
    print(f"Annotation files in '{remote_annos_path}': {count}")
    return remote_annos_path, count


def download_random_pair(host, username, project_path, local_dir):
    remote_base = f"/mnt/bfd/datasets/autobaans/3dod/{project_path}"
    remote_annos = f"{remote_base}/annos"
    remote_pcs = f"{remote_base}/pcs"

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Default policy is RejectPolicy: only connect to hosts in ~/.ssh/known_hosts
    client.connect(hostname=host, username=username)

    # List files (without full paths)
    stdin, stdout, stderr = client.exec_command(f"ls {remote_annos}")
    files = [line.strip() for line in stdout.readlines()]

    if not files:
        print("No annotation files found.")
        client.close()
        return None, None

    # Pick a random filename
    filename = random.choice(files)
    remote_anno = f"{remote_annos}/{filename}"
    remote_pc = f"{remote_pcs}/{filename.replace('.pickle', '.npy.npz')}"

    local_anno = local_dir / filename
    local_pc = local_dir / os.path.basename(remote_pc)

    with SCPClient(client.get_transport()) as scp:
        print(f"Downloading: {filename}")
        scp.get(remote_anno, local_path=str(local_anno))
        scp.get(remote_pc, local_path=str(local_pc))

    client.close()
    return local_pc, local_anno


def load_pointcloud(pc_path):
    """Load point cloud from .npy.npz file."""
    try:
        with np.load(pc_path, allow_pickle=True, mmap_mode="r") as data:
            pointcloud = data["arr_0"]
        pointcloud = np.c_[
            pointcloud[:, 0],
            pointcloud[:, 1],
            pointcloud[:, 2],
            pointcloud[:, 3] / 2**16,
        ]
        return pointcloud
    except Exception as e:
        print("Error loading pointcloud from:", pc_path, "ERR:", e)
        return None


def load_cuboids(anno_path):
    """Load cuboid annotations from pickle file."""
    try:
        with open(anno_path, "rb") as f:
            annotations = pickle.load(f)

        cuboids = [
            Cube3D(
                scale=obj["scale"],
                coordinates=obj["coordinates"],
                rotation=obj["rotation"],
            )
            for obj in annotations
        ]
        # for obj in annotations:
        # print(obj)
        return cuboids
    except Exception as e:
        print("Error loading annotations from:", anno_path, "ERR:", e)
        return None


def create_cuboid_mesh(cuboid, color=[1, 0, 0]):
    """Create an Open3D mesh for a cuboid using Cube3D corner points."""
    corners = cuboid.corners().T  # Transpose from (3,8) to (8,3)
    triangles = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def visualize_pointcloud_with_cuboids(pc_path, anno_path):
    """Visualize point cloud with cuboid annotations."""
    print(f"Visualizing {pc_path.name}")
    points = load_pointcloud(pc_path)
    if points is None:
        print("Failed to load point cloud")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ only
    pcd.paint_uniform_color([0.8, 0.8, 0.8])

    cuboids = load_cuboids_from_olf("openlabel-cartesian.json", 23012)
    print(f"Found {len(cuboids)} cuboids in cartesian OLF")
    closest, dist = find_closest_cuboid(cuboids)
    print(f"Closest cuboid: {closest.coordinates} with distance {dist}")
    cuboids = load_cuboids_from_olf("openlabel.json", 23012)
    print(f"Found {len(cuboids)} cuboids in OLF")
    closest, dist = find_closest_cuboid(cuboids)
    print(f"Closest cuboid: {closest.coordinates} with distance {dist}")
    cuboids = load_cuboids(anno_path)
    print(f"Found {len(cuboids)} cuboids in DB")
    closest, dist = find_closest_cuboid(cuboids)
    print(f"Closest cuboid: {closest.coordinates} with distance {dist}")
    if cuboids is None:
        print("Failed to load annotations")
        return

    geometries = [pcd] + [create_cuboid_mesh(cuboid) for cuboid in cuboids if cuboid]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualization", width=1024, height=768)
    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.run()
    vis.destroy_window()


def test_visualization():
    """Find and visualize the first available point cloud with annotations."""
    pc_files = list(SCRIPT_DIR.glob("*.npy.npz"))

    if not pc_files:
        print("No point cloud files found in the directory.")
        return

    for pc_path in pc_files:
        print(f"Trying point cloud file: {pc_path}")
        anno_path = SCRIPT_DIR / pc_path.name.replace(".npy.npz", ".pickle")
        print(f"Looking for annotation file: {anno_path}")

        if anno_path.exists():
            visualize_pointcloud_with_cuboids(pc_path, anno_path)
            return
        print("No matching annotation found, retrying...")

    print("No valid point cloud and annotation pair found.")


def get_cube_from_olf(cube):
    # position is first 3 values
    position = cube[0:3]
    # rotation is next 4 values
    rotation = cube[3:7]
    # scale is last 3 values
    scale = cube[7:10]
    cuboid = Cube3D(scale, rotation, position)
    return cuboid


def find_closest_cuboid(cuboids):
    min_distance = 100000
    closest_cuboid = None
    for cuboid in cuboids:
        distance = np.linalg.norm(cuboid.coordinates)
        if distance < min_distance:
            min_distance = distance
            closest_cuboid = cuboid
    return closest_cuboid, min_distance


def load_cuboids_from_olf(olf, timestamp=None):
    """
    Extracts cuboids from the OLF file.
    :param olf: OpenLabelFormat dictionary
    :return: List of Cube3D objects
    """
    # load olf file
    with open(olf) as f:
        olf = json.load(f)
    cuboids = []
    for frame in olf["openlabel"]["frames"]:
        if timestamp is not None:
            if (
                olf["openlabel"]["frames"][frame]["frame_properties"]["timestamp"]
                != timestamp
            ):
                continue
        for object in olf["openlabel"]["frames"][frame]["objects"]:
            if (
                "cuboid"
                in olf["openlabel"]["frames"][frame]["objects"][object][
                    "object_data"
                ].keys()
            ):
                cuboid = olf["openlabel"]["frames"][frame]["objects"][object][
                    "object_data"
                ]["cuboid"][0]
                cuboid_obj = Cube3D(
                    scale=cuboid["val"][7:10],
                    coordinates=cuboid["val"][0:3],
                    rotation=cuboid["val"][3:7],
                )
                cuboids.append(cuboid_obj)
    return cuboids


if __name__ == "__main__":
    reachable_host = check_ssh_connection(SSH_HOST, SSH_USERNAME)
    if reachable_host:
        chosen_project = list_remote_directories(reachable_host, SSH_USERNAME)
        _, _ = count_anno_files(reachable_host, SSH_USERNAME, chosen_project)
        local_dir = SCRIPT_DIR / "downloaded"
        local_dir.mkdir(exist_ok=True)
        pc_path, anno_path = download_random_pair(
            reachable_host, SSH_USERNAME, chosen_project, local_dir
        )
        if pc_path and anno_path:
            visualize_pointcloud_with_cuboids(pc_path, anno_path)
    # test_visualization()
    # load openlabel.json
    # with open("openlabel.json") as f:
    #    data = json.load(f)
    # print(data["openlabel"]["frames"]["1"]["frame_properties"]["timestamp"])

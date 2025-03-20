import random
import numpy as np
import open3d as o3d
import pickle
import os
from pathlib import Path
from kognic.judgement_shapes.cube_3d import Cube3D
from scipy.spatial.transform import Rotation

# Set environment variables for OpenGL
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Use OSMesa for software rendering
os.environ['OSMESA_PREFIX'] = '/usr'  # Adjust this path if needed

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
        print("Error loading pointcloud from: ", pc_path, "ERR:", e)
        return None

def load_cuboids(anno_path):
    """Load cuboid annotations from pickle file."""
    try:
        with open(anno_path, "rb") as f:
            annotations = pickle.load(f)
        
        # Convert annotations to Cube3D objects
        cuboids = []
        for obj in annotations:
            # Create Cube3D object with the original parameters
            cuboid = Cube3D(
                scale=obj["scale"],
                coordinates=obj["coordinates"],
                rotation=obj["rotation"]
            )
            cuboids.append(cuboid)
        
        return cuboids
    except Exception as e:
        print("Error loading annotations from: ", anno_path, "ERR:", e)
        return None

def create_cuboid_mesh(cuboid, color=[1, 0, 0]):
    """Create an Open3D mesh for a cuboid using Cube3D corner points."""
    # Get corner points from Cube3D and transpose to get (8,3) shape
    corners = cuboid.corners().T  # Transpose from (3,8) to (8,3)
    
    # Define triangles (same as before)
    triangles = np.array([
        [0,1,2], [0,2,3],  # bottom
        [4,5,6], [4,6,7],  # top
        [0,1,5], [0,5,4],  # sides
        [1,2,6], [1,6,5],
        [2,3,7], [2,7,6],
        [3,0,4], [3,4,7]
    ])
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    
    return mesh

def visualize_pointcloud_with_cuboids(pc_path, anno_path):
    """Visualize point cloud with cuboid annotations"""
    print(f"Visualizing {pc_path.name}")
    
    # Load point cloud using our specialized function
    points = load_pointcloud(pc_path)
    if points is None:
        print("Failed to load point cloud")
        return
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Only use XYZ coordinates
    
    # Set point cloud color to light gray
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    
    # Load annotations
    with open(anno_path, 'rb') as f:
        annotations = pickle.load(f)
    
    # Create visualization geometries
    geometries = [pcd]
    
    # Add cuboids for each annotation
    for anno in annotations:
        # Create Cube3D object with the annotation parameters
        cuboid = Cube3D(
            scale=anno["scale"],
            coordinates=anno["coordinates"],
            rotation=anno["rotation"]
        )
        mesh = create_cuboid_mesh(cuboid)
        if mesh is not None:
            geometries.append(mesh)
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Visualization", width=1024, height=768)
    
    # Add geometries to the visualizer
    for geometry in geometries:
        vis.add_geometry(geometry)
    
    # Set default camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 1, 0])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 0, 1])
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def test_visualization():
    """Test function to visualize a single point cloud with annotations."""
    base_path = Path("/mnt/bfd/datasets/autobaans/3dod/conti3d")
    pc_dir = base_path / "pcs"
    anno_dir = base_path / "annos"
    
    for pc_path in pc_dir.glob("*.npy.npz"):
        print(f"Trying point cloud file: {pc_path}")
        # Get the base name without both extensions and add .pickle
        base_name = pc_path.name.replace('.npy.npz', '.pickle')
        anno_path = anno_dir / base_name
        print(f"Looking for annotation file: {anno_path}")
        
        if anno_path.exists():
            break
        print("No matching annotation found, retrying...")
    
    print(f"Visualizing {pc_path.name}")
    visualize_pointcloud_with_cuboids(pc_path, anno_path)  # Pass Path objects directly

if __name__ == "__main__":
    test_visualization() 
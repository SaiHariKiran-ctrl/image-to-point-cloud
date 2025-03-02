import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def view_xyz_file(xyz_path):
    """
    View a .xyz point cloud file using Open3D
    """
    points = np.loadtxt(xyz_path, delimiter=' ')
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    z_values = points[:, 2]
    colors = plt.cm.viridis((z_values - z_values.min()) / (z_values.max() - z_values.min()))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, coordinate_frame],
                                    window_name="Point Cloud Viewer",
                                    width=1024,
                                    height=768,
                                    left=50,
                                    top=50,
                                    point_show_normal=False,
                                    mesh_show_wireframe=False,
                                    mesh_show_back_face=False)

def analyze_point_cloud(xyz_path):
    """
    Print basic statistics about the point cloud
    """
    points = np.loadtxt(xyz_path, delimiter=' ')
    print(f"Number of points: {len(points)}")
    print(f"Bounding box:")
    print(f"  Min: ({points[:, 0].min():.3f}, {points[:, 1].min():.3f}, {points[:, 2].min():.3f})")
    print(f"  Max: ({points[:, 0].max():.3f}, {points[:, 1].max():.3f}, {points[:, 2].max():.3f})")
    print(f"Center: ({points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f})")

if __name__ == "__main__":
    # Replace with your .xyz file path
    xyz_file = "view_1_generated.xyz"
    
    print("Point Cloud Statistics:")
    print("-" * 20)
    analyze_point_cloud(xyz_file)
    print("\nOpening point cloud viewer...")
    view_xyz_file(xyz_file)

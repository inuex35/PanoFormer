import OpenEXR
import Imath
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import json
import os

def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel = exr_file.channel('R', float_type)
    depth_map = np.frombuffer(channel, dtype=np.float32).reshape(size[1], size[0])
    return depth_map

def read_color_image(file_path):
    color_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    return color_image

def euler_to_rotation_matrix(angles, order="XYZ"):
    roll, pitch, yaw = angles
    roll = roll
    pitch = - (pitch - np.pi)
    yaw = (yaw - np.pi)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    if order == "XYZ":
        R = Rz @ Ry @ Rx
    elif order == "XZY":
        R = Ry @ Rz @ Rx
    elif order == "YXZ":
        R = Rz @ Rx @ Ry
    elif order == "YZX":
        R = Rx @ Rz @ Ry
    elif order == "ZXY":
        R = Ry @ Rx @ Rz
    elif order == "ZYX":
        R = Rx @ Ry @ Rz
    else:
        raise ValueError("Invalid order. Supported orders are: XYZ, XZY, YXZ, YZX, ZXY, ZYX.")
    
    return R

def transform_point_cloud(point_cloud, angles, T, rotation_order, translation_order):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    transformation = np.eye(4)

    transformation[:3, :3] = euler_to_rotation_matrix(angles, rotation_order)
    
    T = np.array(T)
    T[0] = - T[0]
    # T[1] = - T[1]
    T[2] = - T[2]

    if translation_order == "XYZ":
        transformation[:3, 3] = T
    elif translation_order == "XZY":
        transformation[:3, 3] = T[[0, 2, 1]]
    elif translation_order == "YXZ":
        transformation[:3, 3] = T[[1, 0, 2]]
    elif translation_order == "YZX":
        transformation[:3, 3] = T[[1, 2, 0]]
    elif translation_order == "ZXY":
        transformation[:3, 3] = T[[2, 0, 1]]
    elif translation_order == "ZYX":
        transformation[:3, 3] = T[[2, 1, 0]]
    else:
        raise ValueError("Invalid order. Supported orders are: XYZ, XZY, YXZ, YZX, ZXY, ZYX.")
    print(transformation)

    pcd.transform(transformation)

    return np.asarray(pcd.points)

def point_cloud_to_reconstruction_json(points, colors, angles, translation, point_start_id, rotation_order, translation_order):
    transformed_points = transform_point_cloud(points, angles, translation, rotation_order, translation_order)

    points_dict = {}
    for i, (point, color) in enumerate(zip(transformed_points, colors), start=point_start_id):
        points_dict[str(i)] = {
            "coordinates": point.tolist(),
            "color": (color * 255).astype(int).tolist()
        }

    return points_dict

def depth_to_point_cloud_equirectangular(depth_map, color_image, sample_step=10):
    height, width = depth_map.shape
    points = []
    colors = []

    for y in range(0, height, sample_step):
        for x in range(0, width, sample_step):
            z = 0.2 / (depth_map[y, x] ** 2.220002)
            if z > 100:
                continue
            theta = 2 * np.pi * (x / width - 0.5)
            phi = np.pi * (0.5 - y / height)

            X = z *  np.sin(theta) * np.cos(phi)
            Y = - z * np.sin(phi)
            Z = z *  np.cos(theta) * np.cos(phi)

            points.append([X, Y, Z])
            colors.append(color_image[y, x] / 255.0)  # Normalize color values

    return np.array(points), np.array(colors)

class CustomVisualizer:
    def __init__(self, point_cloud, colors):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.pcd)
        self.translate_step = 1
        self.setup_callbacks()

    def setup_callbacks(self):
        self.vis.register_key_callback(ord("W"), self.translate_forward)
        self.vis.register_key_callback(ord("S"), self.translate_backward)
        self.vis.register_key_callback(ord("A"), self.translate_left)
        self.vis.register_key_callback(ord("D"), self.translate_right)
        self.vis.register_key_callback(ord("Q"), self.translate_up)
        self.vis.register_key_callback(ord("E"), self.translate_down)

    def translate_forward(self, vis):
        self.pcd.translate([0, 0, -self.translate_step])
        vis.update_geometry(self.pcd)

    def translate_backward(self, vis):
        self.pcd.translate([0, 0, self.translate_step])
        vis.update_geometry(self.pcd)

    def translate_left(self, vis):
        self.pcd.translate([-self.translate_step, 0, 0])
        vis.update_geometry(self.pcd)

    def translate_right(self, vis):
        self.pcd.translate([self.translate_step, 0, 0])
        vis.update_geometry(self.pcd)

    def translate_up(self, vis):
        self.pcd.translate([0, self.translate_step, 0])
        vis.update_geometry(self.pcd)

    def translate_down(self, vis):
        self.pcd.translate([0, -self.translate_step, 0])
        vis.update_geometry(self.pcd)

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

def show_depth_image(depth_map, scale_factor=0.1):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Adjust x and y according to the scale factor
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            if 0 <= original_x < depth_map.shape[1] and 0 <= original_y < depth_map.shape[0]:
                depth_value = depth_map[original_y, original_x]
                display_image = depth_image_color.copy()
                cv2.putText(display_image, f"Depth: {depth_value:.4f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(display_image, f"Dist: {1/depth_value:.4f}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imshow('Depth Map', display_image)

    # Resize the depth map
    depth_map_resized = cv2.resize(depth_map, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Normalize the resized depth map for display
    depth_image = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = np.uint8(depth_image)
    
    # Convert the depth image to color
    depth_image_color = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    
    cv2.imshow('Depth Map', depth_image_color)
    cv2.setMouseCallback('Depth Map', mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_inverse_depth_middle_y(depth_map):
    height, width = depth_map.shape
    middle_x = width // 2
    depth_values = depth_map[:, middle_x]
    inverse_depth_values = 1.0 / depth_values
    pitch_angles = np.linspace(-np.pi/2, np.pi/2, height) * 180 / np.pi  # ピッチ角の範囲を指定

    plt.figure()
    plt.plot(pitch_angles, inverse_depth_values)
    plt.xlabel('Pitch Angle (radians)')
    plt.ylabel('Inverse Depth (1/Depth)')
    plt.title('Inverse Depth Profile along Middle Y-Axis')
    plt.grid(True)
    plt.show()

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main(depth_folder, color_folder, reconstruction_file, new_reconstruction_file):
    json_data = load_json(reconstruction_file)
    color_files = os.listdir(color_folder)

    for color_file_name in color_files:
        corresponding_shot = None
        corresponding_shot = json_data[0]["shots"].get(color_file_name)

        if corresponding_shot is not None:
            depth_file_name = color_file_name.replace('FinalColor.png', 'SceneDepth.exr')
            depth_file_path = os.path.join(depth_folder, depth_file_name)
            color_file_path = os.path.join(color_folder, color_file_name)

            depth_map = read_exr(depth_file_path)
            color_image = read_color_image(color_file_path)

            print(f"Processing {depth_file_name} and {color_file_name}")
            points, colors = depth_to_point_cloud_equirectangular(depth_map, color_image, sample_step=50)
            translation = corresponding_shot["translation"]
            rotation = corresponding_shot["rotation"]

            point_start_id = max(int(k) for k in json_data[0]["points"].keys()) + 1 if json_data[0]["points"] else 0
            new_points = point_cloud_to_reconstruction_json(points, colors, rotation, translation, point_start_id, "ZYX", "XZY")
            print("Orders are: XYZ, XZY, YXZ, YZX, ZXY, ZYX.")

            json_data[0]["points"].update(new_points)
            save_json(json_data, new_reconstruction_file)

if __name__ == "__main__":
    depth_folder = 'SceneDepth'  # EXRファイルのフォルダを指定
    color_folder = 'FinalColor'  # カラー画像のフォルダを指定
    reconstruction_file = 'reconstruction.json'  # JSONファイルのパスを指定
    new_reconstruction_file = 'new_reconstruction.json'  # 新しいJSONファイルのパスを指定
    main(depth_folder, color_folder, reconstruction_file, new_reconstruction_file)

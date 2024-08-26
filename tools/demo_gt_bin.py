import argparse
import os
import numpy as np
import json
import open3d as o3d

from pcdet.utils import common_utils
from visual_utils.open3d_box import create_box
from visual_utils.open3d_arrow import create_arrow

def create_coordinate(size=2.0, origin=[0, 0, 0]):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0]
    )
    return mesh_frame


def create_box_with_arrow(box, color=None):
    """
    box: list(8) [ x, y, z, dx, dy, dz, yaw]
    """
    box_o3d = create_box(box, color)
    x = box[0]
    y = box[1]
    z = box[2]
    l = box[3]
    yaw = box[6]
    # get direction arrow
    dir_x = l / 2.0 * np.cos(yaw)
    dir_y = l / 2.0 * np.sin(yaw)

    arrow_origin = [x - dir_x, y - dir_y, z]
    arrow_end = [x + dir_x, y + dir_y, z]
    arrow = create_arrow(arrow_origin, arrow_end, color)

    return box_o3d, arrow


def draw_points_with_boxes(points, gt_boxes):
    """
    points: (N, 4) [x, y, z, intensity]
    gt_boxes: (N, 7) np.array = n*7  ( x, y, z, dx, dy, dz, yaw)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # create points
    # color_bin = (points[:, 3].max() - points[:, 3].min()) / 255.0
    # colors = (points[:, 3] - points[:, 3].min()) / color_bin / 255.0
    # points_color = [[0.5*c, 0.5*c, 0.5*c] for c in colors]
    points_color = [[0.5, 0.5, 0.5]]  * points.shape[0]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:,:3])
    pc.colors = o3d.utility.Vector3dVector(points_color)
    vis.add_geometry(pc)

    # create boxes with colors with arrow
    gt_box_color = [0.4,0.7,0.2]
    if gt_boxes is not None:
        gts_o3d = []
        for i in range(gt_boxes.shape[0]):
            box = gt_boxes[i]
            gt_o3d, arrow = create_box_with_arrow(box, gt_box_color)
            gts_o3d.append(gt_o3d)
            gts_o3d.append(arrow)
        [vis.add_geometry(element) for element in gts_o3d]

    # coordinate frame
    coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='data/caic_lidar/pc_pro/sample1/bin/1716688203.511341.bin')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of point cloud data file.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = common_utils.create_logger()
    logger.info('--------- Visualize GT of the Point Cloud Data ---------')
    data_path = args.data_path
    tmp = data_path.split('/')
    basename = tmp[-1].split('.bin')[0]
    gt_path = os.path.join('/'.join(tmp[:-2]), 'label', basename+'.json')

    points = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)
    gt_boxes = json.load(open(gt_path, 'r'))['labels']
    gt_npy = []
    for gtb in gt_boxes:
        gt_npy.append([gtb['center']['x'], gtb['center']['y'], gtb['center']['z'], gtb['size']['x'], \
                       gtb['size']['y'], gtb['size']['z'], gtb['rotation']['yaw']])
    gt_boxes = np.array(gt_npy)

    draw_points_with_boxes(points, gt_boxes)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
"""
Extract point cloud and images from the raw data, reorg the directories
PCD是自车坐标系， FLU
"""
import os
import shutil
import open3d as o3d
import numpy as np
import argparse
import json
import cv2
import tqdm

from pathlib import Path
from pcdet.utils.pose import Pose
from pyquaternion import Quaternion
from pcd_utils import *
from pcdet.utils.vis import draw_pts_on_img


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--src_dir', type=str, default='data/caic_bev/sample_raw')
    parser.add_argument('--data_dir', type=str, default='data/caic_bev/sample_proc')
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--nocopy', action='store_true')
    parser.add_argument('--nopoints', action='store_true')
    parser.add_argument('--pcd_topic', type=str, default='/perception/pcd_in_map')
    parser.add_argument('--ego_pcd', action='store_true')
    parser.add_argument('--ego_zoffset', type=float, default=0)
    parser.add_argument('--draw_count', type=int, default=40)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pcd_paths = []
    args = parse_args()

    src_dir = Path(args.src_dir)
    data_dir = Path(args.data_dir)

    for sub in ['front_wide', 'front_right', 'front_left', 'rear', 'rear_right', 'rear_left', 'data', \
                'front_wide_ex', 'front_right_ex', 'front_left_ex', 'rear_ex', 'rear_right_ex', 'rear_left_ex', 'lidar_ex']:
        sub_dir = data_dir / sub
        sub_dir.mkdir(exist_ok=True, parents=True)
    pcd_dir = data_dir / 'pcd'
    pcd_dir.mkdir(exist_ok=True)

    for root, parent, files in os.walk(src_dir):
        if len(parent) > 0: continue
        for file in files:
            if not file.endswith('.pcd'): continue
            pcd_path = Path(root) / file
            pcd_paths.append(pcd_path)
    
    count = 0
    for pcd_path in tqdm.tqdm(pcd_paths):
        pcd_file = open(pcd_path, 'rb')
        raw_pc = point_cloud_from_fileobj(pcd_file)
        points = raw_pc.reshape(-1, 4).copy()
        basename = os.path.basename(pcd_path).split('.pcd')[0]
        bin_path = data_dir / 'bin'
        bin_path.mkdir(exist_ok=True, parents=True)
        print('Intensity min max mean: ', points[:, 3].min(), points[:, 3].max(), points[:, 3].mean())
        points.tofile(bin_path / (basename+'.bin'))

        if not args.nocopy:
            try:
                shutil.copy(str(pcd_path), str(pcd_dir))
            except:
                pass
        
        info_path = pcd_path.parent.parent / 'data' / (basename+'.json')
        info = json.load(open(info_path.__str__(), 'r'))
        if 'TX_V_M' in info['sensors'][args.pcd_topic]:
            v2world = Pose.from_matrix(np.array(info['sensors'][args.pcd_topic]['TX_V_M']))
            mat = v2world.matrix.tolist()
        else:
            mat = np.eye(4, dtype=np.float32).tolist()
        pose_ex = {"ex": mat, "timestamp": float(info['sensors'][args.pcd_topic]['time'])}
        json.dump(pose_ex, open(str(data_dir / 'lidar_ex' / (basename+'.json')),'w'))

        for cam in ['front_wide', 'front_right', 'front_left', 'rear', 'rear_right', 'rear_left']:
            raw_topic = '/{}/image_raw'.format(cam)
            compress_topic = '/{}/image_raw/compressed'.format(cam)
            if raw_topic in info['sensors']:
                cam_info = info['sensors'][raw_topic]
            elif compress_topic in info['sensors']:
                cam_info = info['sensors'][compress_topic]
            else:
                continue
            sub_dir = data_dir / cam
            img_name = os.path.basename(cam_info['file'])
            img_path = str(pcd_path.parent.parent / cam / img_name)

            #======================================================================================================
            # if not args.nocopy:
            #     shutil.copy(img_path, str(sub_dir/(basename+'.jpg'))) # 图片重命名成lidar时间戳
            #------------------------------------------------------------------------------------------------------
            intrinsics = np.array(cam_info['intrinsics'])
            distortion = np.array(cam_info['D'])
            img_dist = cv2.imread(img_path)
            # img_ud = cv2.undistort(img_dist, intrinsics, distortion)
            # img_ud = cv2.fisheye.undistortImage(img_dist, intrinsics, distortion)
            size = (img_dist.shape[1],img_dist.shape[0])
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, distortion, np.eye(3), intrinsics, size, cv2.CV_16SC2)
            img_ud = cv2.remap(img_dist, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cv2.imwrite(str(sub_dir/(basename+'.jpg')), img_ud)
            #======================================================================================================

            # cam_pose = cam_info['pose_slert']['pose'] if 'pose_slert' in cam_info else cam_info['pose_slerp']['pose']
            intr = cam_info['intrinsics']
            k = {'fx': intr[0][0], 'fy': intr[1][1], 'cx': intr[0][2], 'cy': intr[1][2]}
            # cam_pose = Pose(wxyz=np.array([cam_pose[6], *cam_pose[3:6]]), tvec=np.array(cam_pose[:3]))
            v2cam = Pose.from_matrix(np.array(cam_info['Tx']))
            # cam2world = world2cam.inverse()
            # v2cam = world2cam * v2world
            # v2cam = cam2v.inverse()
            cam_ex = {'k': k, 'ex': v2cam.matrix.tolist(), 'timestamp': float(cam_info['time'])}
            json.dump(cam_ex, open(str(data_dir / (cam+'_ex') / (basename+'.json')), 'w'))

            if args.draw and count < args.draw_count:
                # KRT = np.array(intr) @ v2cam.matrix[:3, :]
                # KRT = np.vstack([KRT, np.zeros((1, 4))])
                K = np.zeros((4, 4))
                K[:3, :3] = np.array(intr)
                KRT = K @ v2cam.matrix
                # KRT = K @ v2cam.matrix
                save_path = data_dir / (cam+'_proj') / (basename+'.jpg')
                save_path.parent.mkdir(exist_ok=True)
                # img = cv2.imread(img_path)
                draw_pts_on_img(points[:, :3], img_ud, KRT, str(save_path))
        
        count += 1    
        if not args.nocopy:
            shutil.copy(str(pcd_path.parent.parent / 'data' / (basename+'.json')), str(data_dir / 'data' / (basename+'.json')))
        
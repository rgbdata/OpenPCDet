import os
from pathlib import Path
import json
import numpy as np
import shutil
import re
import tqdm
import struct
import open3d as o3d
import argparse
import lzf

# from pcdet.utils.pose import Pose
from pcd_utils import *
from pcdet.datasets.augmentor import tta_utils
from pcdet.utils.common_utils import extract_label

"""
脚本遍历数据文件夹下所有.pcd文件。
.pcd文件的父级目录同级目录下，寻找同名的.json标签文件。
原始标签移到raw_label文件夹下。
使用tag区分不同处理方式的数据
=============================
原始结构：
xxx/
    pcd/
        xxx.pcd
    label/
        xxx.json

处理后：
xxx/
    pcd/
        xxx.pcd
    raw_label/
        xxx.json

tag/
    bin/
        xxx.bin
    label/
        xxx.json
==============================
"""

def load_pcd(pcd):
    pc = o3d.t.io.read_point_cloud(pcd)
    xyz = pc.point['positions'].numpy()
    intensity = pc.point['intensity'].numpy()
    points = np.hstack([xyz, intensity])
    points = points.astype(np.float32)
    return points


type_map = { 
    'bicycle': 'cyclist',
    'motorcycle': 'cyclist',
    'huge_vehicle': 'big_vehicle',
    # 'tricycle': 'tricycle',
    'tricycle': 'vehicle',
    'car': 'vehicle',
    'bus': 'big_vehicle',
    'truck': 'big_vehicle',
    'construction': 'big_vehicle',
    'deformed_pedestrian': 'pedestrian',
    'bicycle_rider': 'cyclist',
    'motorcycle_rider': 'cyclist',
    'pedestrain': 'pedestrian'
}


def transform_label(label_file, data_dir, zoff=0, yaw=0):
    json_data = json.load(open(label_file, 'r'))
    raw_label_path = label_file.parent.parent / 'raw_label'
    raw_label_path.mkdir(exist_ok=True)
    shutil.move(label_file, raw_label_path / label_file.name)
    for label in json_data['labels']:
        if 'type' not in label: continue
        t = label['type'].lower()
        label['type'] = type_map.get(t, t)
        if 'lidar_3d_bbox' in label:
            for key in label['lidar_3d_bbox']:
                label[key] = label['lidar_3d_bbox'][key]
        box = extract_label(label)
        box = tta_utils.global_rotation(np.array([box]), yaw)[0]
        label['center']['x'] = box[0]
        label['center']['y'] = box[1]
        label['center']['z'] = box[2] + zoff
        label['size']['x'] = box[3]
        label['size']['y'] = box[4]
        label['size']['z'] = box[5]
        label['rotation']['yaw'] = box[6]

    label_file = data_dir / 'label' / label_file.name
    label_file.parent.mkdir(exist_ok=True)
    json.dump(json_data, open(label_file, 'w'))


def parse_args():
    parser = argparse.ArgumentParser(description='arg_parser')
    parser.add_argument('--dst_root', type=str, default='data/caic_lidar/pc_pro')
    parser.add_argument('--tag', type=str, default='raw')
    parser.add_argument('--subs', type=str, default='sample1')
    parser.add_argument('--data_dir', type=str, default='data/caic_lidar')
    parser.add_argument('--zoff', type=float, default=0)
    parser.add_argument('--yaw', type=float, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tag = args.tag
    zoff = args.zoff
    yaw = args.yaw * np.pi
    subs = args.subs.split(',')
    dst_root = Path(args.dst_root)
    dst_root.mkdir(exist_ok=True)
    for sub in subs:
        data_dir = Path(args.data_dir) / sub
        # print(data_dir)
        tag_dir = data_dir / tag
        shutil.rmtree(str(tag_dir), ignore_errors=True)
        for root, parent, files in os.walk(data_dir):
            if len(parent) > 0: continue
            for file in tqdm.tqdm(files):
                if not file.endswith('.pcd'): continue
                pcd_path = Path(root) / file
                candidate_dirs = ['raw_label', 'annotations', 'label', 'json']
                for cand in candidate_dirs:
                    label_file = pcd_path.parent.parent / cand / (file.split('.pcd')[0] + '.json')
                    if label_file.exists():
                        break
                bin_path = tag_dir / 'bin' / (file.split('.pcd')[0] + '.bin')
                bin_path.parent.mkdir(parents=True, exist_ok=True)

                pcd_file = open(pcd_path, 'rb')
                raw_pc = point_cloud_from_fileobj(pcd_file)

                if raw_pc is None:
                    points = load_pcd(str(pcd_path))
                else:
                    points = raw_pc.reshape(-1, 4).copy()

                if label_file.exists():
                    transform_label(label_file, tag_dir, zoff, yaw)
                
                vmask = ~np.any(np.isnan(points) | np.isinf(points), axis=-1)
                points = points[vmask]
                points = np.clip(points, -255, 255)
                points[:, 2] += zoff
                points = tta_utils.global_rotation(points, yaw)
                points.tofile(open(bin_path, 'wb'))
        try:
            shutil.rmtree(str(dst_root/sub), ignore_errors=True)
        except:
            pass
        shutil.move(str(tag_dir), str(dst_root/sub))



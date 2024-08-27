import os
from pathlib import Path
import json
import numpy as np
import shutil
import re
import tqdm
import struct
import open3d as o3d
import lzf

# from pcdet.utils.common_utils import extract_label
# from pcdet.datasets.augmentor import tta_utils


numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)

def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.
    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in xrange(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_header(lines):
    """
    Parse header of PCD files
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            print("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def parse_binary_pc_data(f, metadata):
    rowstep = metadata['points'] * 4 * 4
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.frombuffer(buf, dtype=np.float32)


def parse_uninorm(f, metadata, size):
    point_size = sum(size)
    num = metadata['points']
    pts = np.zeros((num, 4), dtype=np.float32)
    for i in range(num):
        point = f.read(point_size)
        x = struct.unpack('f', point[:4])[0]
        y = struct.unpack('f', point[4:8])[0]
        z = struct.unpack('f', point[8:12])[0]
        intensity = struct.unpack('B', point[12:13])[0]
        pts[i, 0] = x
        pts[i, 1] = y
        pts[i, 2] = z
        pts[i, 3] = float(intensity)
    return pts


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size =\
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros((metadata['width'], 4), dtype=np.float32)
    ix = 0
    for dti in range(min(4, len(dtype))):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.frombuffer(buf[ix:(ix+bytes)], dt)
        # pc_data[dtype.names[dti]] = column
        pc_data[:, dti] = column
        ix += bytes
    return pc_data


def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip().decode('utf8')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    size = dtype.itemsize
    pc_data = None
    if metadata['data'] == 'binary':
        if size == 16:
            pc_data = parse_binary_pc_data(f, metadata)
        else:
            pc_data = parse_uninorm(f, metadata, size)
            # pc = o3d.io.read_point_cloud(str(pcd_path))
            # points = np.asarray(pc.points)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"', metadata['data'])
    return pc_data
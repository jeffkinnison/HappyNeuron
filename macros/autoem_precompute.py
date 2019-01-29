'''Example Usage.
raw_to_precomputed \
--image /mnt/md0/EM/SNEMI/train-input.tif \
--labels /mnt/md0/EM/SNEMI/train-labels.tif \ 
--precomputed /mnt/md0/EM/SNEMI/train-precomputed \
--multi \
--scale 3 \
--large False \
'''
from __future__ import print_function
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py
import neuroglancer
import dxchange
import argparse
from pprint import pprint
import os, signal
import re
from ast import literal_eval as make_tuple
from .reader import check_stack_len,omni_read 
import json
from scipy.ndimage.interpolation import zoom
from cloudvolume import CloudVolume
from tqdm import tqdm

logger = logging.getLogger(__name__)
def build_pyramid_info(info, scale_up_to=3):
    if scale_up_to == 0: return info
    scale_0 = info['scales'][0]
    new_resolution = scale_0['resolution']
    new_size = scale_0['size']
    for i in range(1,scale_up_to):
        new_resolution = [r*2 for r in new_resolution]
        new_size = [s//2 for s in new_size]
        new_scale = {
            "encoding": scale_0['encoding'],
            "chunk_sizes": scale_0['chunk_sizes'],
            "key": "_".join(map(str, new_resolution)),
            "resolution": list(map(int, new_resolution)),
            "voxel_offset": list(map(int, scale_0['voxel_offset'])),
            "size": list(map(int, new_size))
        }



        info['scales'].append(new_scale)
    return info

def local_to_cloud(data, cloud_path, layer_type=None, resolution=None, scale=0):
    '''currently support 
        layer_type: 'image' or 'segmentation'
        resolution: tuple of 3 '''
    data = np.moveaxis(data, 0, 2)
    if not os.path.exists(cloud_path):
        os.makedirs(cloud_path)

    info = CloudVolume.create_new_info(1, 
            layer_type=layer_type, 
            data_type=str(data.dtype), 
            encoding='raw',
            resolution=list(resolution),
            voxel_offset=(0,0,0),
            volume_size=data.shape,
            )

    info = build_pyramid_info(info, scale)

    if layer_type == 'segmentation':
        info['mesh'] = 'mesh'
    with open(os.path.join(cloud_path, 'info'), 'w') as f:
        json.dump(info, f)


    for i in range(0,scale):    
        vol = CloudVolume('file://'+cloud_path, mip=i,compress='') # Basic Example
        if i > 0:
            data = zoom(data, 0.5)
            x,y,z = vol.volume_size
            data = data[0:x, 0:y, 0:z]
        print(vol.volume_size, data.shape)
        vol[:,:,:] = data
def large_data_generator(stack_name, begin=0, end=64, step=64, dtype=None, multi=False):
    for i in range(begin, end, step):
        if i+step > end:
            data = dxchange.read_tiff_stack(stack_name, ind=range(i,end))
        else:
            data = dxchange.read_tiff_stack(stack_name, ind=range(i,i+step))
        if not multi and dtype=='uint32':
            data = np.nan_to_num(data>0)
        if dtype:
            data = data.astype(dtype)
        data = np.moveaxis(data, 0, 2)
        yield (i,data)
    pass


def large_local_to_cloud(data_path, cloud_path, begin=None, end=None, dtype=None, multi=False,z_step=64,
         layer_type=None, chunk_size=(64,64,64), resolution=None, scale=0):
    ''' 
    when data is a tiffstack above RAM limit 
    layer_type: 'image' or 'segmentation'
    resolution: tuple of 3 
    '''
    if not begin and not end:
        S,L = check_stack_len(data_path) # start and length
    else:
        S,L = begin, end-begin
    print(S,L)

    first_slice = dxchange.read_tiff(data_path)
    X,Y = first_slice.shape
    #volume_size = (L,X,Y)
    volume_size = (X,Y,L)
    if not dtype:
        data_type = first_slice.dtype
    else:
        data_type = dtype

    data_generator = large_data_generator(data_path, S, S+L, z_step, data_type, multi)

    if not os.path.exists(cloud_path):
        os.makedirs(cloud_path)

    info = CloudVolume.create_new_info(1, 
            layer_type=layer_type, 
            data_type=str(data_type), 
            encoding='raw',
            chunk_size=chunk_size,
            resolution=list(resolution),
            voxel_offset=(0,0,0),
            volume_size=volume_size,
            )

    info = build_pyramid_info(info, scale)
    if layer_type == 'segmentation':
        info['mesh'] = 'mesh'


    pprint(info)
    with open(os.path.join(cloud_path, 'info'), 'w') as f:
        json.dump(info, f)
    print('>>', L, z_step)
    for i,data in tqdm(data_generator, total=L//z_step):
        curr_z_start = i
        curr_z_step = z_step
        for j in range(0,scale):    
            vol = CloudVolume('file://'+cloud_path, mip=j,compress='') # Basic Example
            x,y,z = vol.volume_size
            
            if j == 0 and i+curr_z_step >= z:
                curr_z_step = z-curr_z_start
            if j > 0:
                data = data[::2,::2,::2]
                data = data[0:x, 0:y, 0:z]
            
            vol[:,:,curr_z_start:curr_z_start+curr_z_step] = data[:,:,:curr_z_step]
            curr_z_start //= 2
            curr_z_step //= 2


        
    return
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', default=None)
    parser.add_argument( '--labels', default=None)
    parser.add_argument( '--precomputed', default='./precomputed')
    parser.add_argument( '--multi', default=False)
    parser.add_argument( '--begin', type=int, default=None)
    parser.add_argument( '--end', type=int, default=None)
    parser.add_argument( '--resolution', type=str, default='10,10,10')
    parser.add_argument( '--scale', type=int, default=3)
    parser.add_argument( '--large', default=True)
    parser.add_argument( '--chunk_size', type=str, default='64,64,64')
    parser.add_argument( '--z_step', type=int, default=None)


    
    args = parser.parse_args()
    resolution = tuple(int(d) for d in args.resolution.split(','))
    chunk_size = tuple(int(d) for d in args.chunk_size.split(','))
    if not args.z_step:
        z_step = int(chunk_size[0]) * 2 ** (int(args.scale)-1)
        print("z_step:{}".format(z_step))
    else:
        z_step = args.z_step

    if args.image is not None: 
        image_cloud_path = os.path.join(args.precomputed, 'image')
        if not args.large:
            image = omni_read(args.image, args.begin, args.end)
            local_to_cloud(image, image_cloud_path, layer_type='image', resolution=resolution, scale=args.scale)
        else:
            large_local_to_cloud(args.image, image_cloud_path, begin=args.begin, end=args.end, chunk_size=chunk_size, z_step=z_step,
                layer_type='image', resolution=resolution, scale=args.scale)


    if args.labels is not None: 
        labels_cloud_path = os.path.join(args.precomputed, 'labels')
        if not args.large:
            labels = omni_read(args.labels, args.begin, args.end)
            if not args.multi:
                labels = np.uint32(np.nan_to_num(labels)>0)
            else:
                labels = np.uint32(np.nan_to_num(labels))
            print(labels.shape, labels.dtype)
            local_to_cloud(labels, labels_cloud_path, layer_type='segmentation', resolution=resolution, scale=args.scale)
        else:
            large_local_to_cloud(args.labels, labels_cloud_path, begin=args.begin, end=args.end, dtype='uint64', multi=args.multi, z_step=z_step,
                layer_type='segmentation', resolution=resolution, scale=args.scale)
            
    



if __name__ == '__main__':
    main()








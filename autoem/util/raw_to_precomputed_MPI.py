#!/usr/env/bin python3
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


try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
except:
    class pseudo_comm():
        def __init__(self):
            pass
        def Barrier(self):
            pass
        def send(self, placeholder, **kwargs):
            pass
        def recv(self, **kwargs):
            pass
    mpi_comm = pseudo_comm()
    mpi_rank = 0
    mpi_size = 1


def allocate_mpi_subsets_cont_chunk(n_task, size):
    task_list = range(n_task)
    sets = []
    per_rank_floor = int(np.floor(n_task/size))
    remainder = n_task % size
    start = 0

    for i in range(size):
        length = per_rank_floor
        if remainder > 0:
            length += 1
            remainder -= 1
        sets.append(task_list[start:start+length])
        start = start + length
    return sets


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

def large_local_to_cloud(data_path, cloud_path, begin=None, end=None, dtype=None, multi=False,z_step=64,
         layer_type=None, chunk_size=(64,64,64), resolution=None, scale=0):
    ''' when data is a tiffstack above RAM limit 
        layer_type: 'image' or 'segmentation'
        resolution: tuple of 3 '''
    
    if not begin and not end:
        S,L = check_stack_len(data_path) # start and length
    else:
        S,L = begin, end-begin
    print(S,L)

    first_slice = dxchange.read_tiff(data_path)
    X,Y = first_slice.shape
    volume_size = (X,Y,L)
    
    if not dtype:
        data_type = first_slice.dtype
    else:
        data_type = dtype
    S_list = gen_S_list[mpi_rank]
    data_generator = large_data_generator(data_path, S_list, S_list+L//mpi_size, z_step, data_type, multi)

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


    for i,data in data_generator:
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

            print(data.shape)
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
    parser.add_argument( '--resolution', type=str, default='(10,10,10)')
    parser.add_argument( '--scale', type=int, default=0)
    parser.add_argument( '--chunk_size', type=str, default='(64,64,64)')
    parser.add_argument( '--z_step', type=int, default=None)


    
    args = parser.parse_args()
    resolution = make_tuple(args.resolution)
    chunk_size = make_tuple(args.chunk_size)

    if not args.z_step:
        z_step = int(chunk_size[0]) * 2 ** (int(args.scale)-1)
        print("z_step:{}".format(z_step))
    else:
        z_step = args.z_step


    if args.image is not None: 
        image_cloud_path = os.path.join(args.precomputed, 'image')
        large_local_to_cloud(args.image, image_cloud_path, begin=args.begin, end=args.end, chunk_size=chunk_size, z_step=z_step,
                layer_type='image', resolution=resolution, scale=args.scale)


    if args.labels is not None: 
        labels_cloud_path = os.path.join(args.precomputed, 'labels')
        large_local_to_cloud(args.labels, labels_cloud_path, begin=args.begin, end=args.end, dtype='uint32', multi=args.multi, z_step=z_step,
                layer_type='segmentation', resolution=resolution, scale=args.scale)
            
    



if __name__ == '__main__':
    main()








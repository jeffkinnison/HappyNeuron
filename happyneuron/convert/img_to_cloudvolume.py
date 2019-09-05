#!/usr/bin/env python3
"""Convert a sequence of images into a CloudVolume archive.

The volume is broken into "layers", or small non-overlapping sub-sequences of
images that can be added to CloudVolume in parallel.

Functions
---------
parse_args
    Parse command line arguments.
load_layer
    Load a sequence of images.
initialize_cloudvolume
    Create a new CloudVolume archive.
write_layer
    Write a layer to CloudVolume.
main
    Create and write data to a new CloudVolume archive.

Dependencies
------------
cloud-volume
dill
mpi4py
numpy
scikit-image
"""

from __future__ import print_function

import argparse
import glob
import logging
import os
import re
import sys

from cloudvolume import CloudVolume
from mpi4py import MPI
import numpy as np
import skimage.io as io


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

LOGGER = logging.getLogger('img_to_cloudvolume.py')
syslog = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)s Rank %(rank)s : %(message)s')
syslog.setFormatter(formatter)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(syslog)
LOGGER = logging.LoggerAdapter(LOGGER, {'rank': str(RANK)})


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser()

    p.add_argument('--input', type=str,
                   help='path to the directory holding the images')
    p.add_argument('--output', type=str,
                   help='path to write the CloudVolume')
    p.add_argument('--mode', type=str, choices=['image', 'segmentation'],
                   default='image', help='write mode for configuring the info file')
    p.add_argument('--ext', type=str, default='.tif',
                   help='extension of the images to load')
    p.add_argument('--resolution', type=int, nargs='*', default=[10, 10, 10],
                   help='resolution of the dataset')
    p.add_argument('--mip', type=int, default=0,
                   help='number of mip levels')
    p.add_argument('--chunk-size', type=int, nargs='*', default=[64, 64, 64],
                   help='size of each CloudVolume block file')
    p.add_argument('--z-step', type=int, default=None)
    p.add_argument('--factor', type=int, nargs='*', default=[2, 2, 2],
                   help='factor to scale between mip levels')
    p.add_argument('--flip-xy', action='store_true',
                   help='pass to transplose the X and Y axes')
    p.add_argument('--memory-limit', type=float, default=10000,
                   help='max memory available to CloudVolume')
    p.add_argument('--offset', type=int, nargs='*', default=[0, 0, 0],
                   help='offset into the volume from the upper-left corner')
    p.add_argument('--quiet', action='store_true',
                   help='pass to disable logging')

    return p.parse_args()


def load_layer(imagelist, z_start, z_end):
    """Load a sequence of images.

    Parameters
    ----------
    imagelist : list of str
        Sorted list of filepaths pointing to images to load.
    z_start : int
        The index of the first image in the layer.

    Returns
    -------
    layer : numpy.ndarray
        The (n, h, w) ndarray containing the n sequential images in the layer.
    """
    # Set up the array for preallocation
    layer = None

    # Load each image in the layer and insert it into the array.
    for i, img in enumerate(imagelist[z_start:z_end]):
        img = io.imread(img)
        if layer is None:
            layer = np.zeros((int(z_end - z_start),) + img.shape,
                             dtype=img.dtype)
        layer[i] += img
    LOGGER.info('Loaded images with shape {}.'.format(layer.shape))
    return layer


def initialize_cloudvolume(path, mode, dtype, resolution, offset, volume_size,
                           chunk_size, mip, factor):
    """Create a new CloudVolume archive.

    Parameters
    ----------
    path : str
        Filepath to the location to write the archive.
    mode : {'image','segmentation'}
        Write mode for configuring the info file.
    dtype : str
        The data type of the images to write.
    resolution : tuple of int
        Imaging resolution of the images in each dimension.
    offset : tuple of int
        Offset within the volume to the start of the archive.
    volume_size : tuple of int
        The dimensions of the volume in pixels.
    chunk_size : tuple of int
        The size of each CloudVolume block in pixels.
    mip : int
        The number of mip levels to include.
    factor : tuple of int
        The factor of change in each dimension across mip levels.

    Returns
    -------
    cv_args : dict
        The parameters needed to re-access the CloudVolume archive.
    """
    # Set the parameters of the info file.
    if mode == 'image':
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type=mode,
            data_type=str(dtype),
            encoding='raw',
            resolution=resolution,
            voxel_offset=offset,
            volume_size=volume_size,
            chunk_size=chunk_size,
            max_mip=0,
            factor=factor
        )
    elif mode == 'segmentation':
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type=mode,
            data_type='uint32',
            encoding='compressed_segmentation',
            resolution=resolution,
            voxel_offset=offset,
            volume_size=list(volume_size),
            chunk_size=chunk_size,
            max_mip=0,
            factor=factor
        )
    else:
        raise ValueError('Cannot write layer of type {}. Must be one of ["image", "segmentation"]')

    # Set up and initialize the CloudVolume object
    cv_args = dict(
        bounded=True, fill_missing=True, autocrop=False,
        cache=False, compress_cache=None, cdn_cache=False,
        progress=False, info=info, provenance=None,
        compress=(mode == 'segmentation'), non_aligned_writes=True,
        parallel=1)

    if mode == 'segmentation':
        for i in range(1, mip + 1):
            info['scales'][i]['compressed_segmentation_block_size'] = \
                info['scales'][0]['compressed_segmentation_block_size']

    cv = CloudVolume(path, mip=0, **cv_args)

    # Create the info file.
    LOGGER.info('Initializing image layer with config {}'.format(cv_args))
    cv.commit_info()
    return cv_args


def write_layer(path, mode, layer, flip_xy, z_start, mip, factor):
    """Write a layer to CloudVolume.

    Parameter
    ---------
    path : str
        Filepath to the location to write the archive.
    layer : numpy.ndarray
        Image data to write to the archive.
    flip_xy : bool
        If True, order ``layer`` as [Y, X, Z]. Otherwise, order ``layer`` as
        [X, Y, Z].
    z_start
        The starting index of ``layer`` within the archive.
    mip
        The number of mip levels to compute.
    factor
        The factor by which to reduce each mip level along each dimension.
    cv_args
        Arguments used to access the CloudVolume archive.
    """
    # Transpose the axes to match the CloudVolume order
    if flip_xy:
        layer = np.transpose(layer, axes=[1, 2, 0])
    else:
        layer = np.transpose(layer, axes=[2, 1, 0])

    cv_args = dict(
        bounded=True, fill_missing=True, autocrop=False,
        cache=False, compress_cache=None, cdn_cache=False,
        progress=False, info=None, provenance=None,
        compress=(mode == 'segmentation'), non_aligned_writes=True,
        parallel=1)

    # Set the volume for each mip level
    for m in range(1):
        LOGGER.info('Writing images {}-{} to MIP level {}'.format(z_start, z_start + layer.shape[-1], mip))
        # Access the CloudVolume
        cv = CloudVolume(path, mip=m, **cv_args)

        # Compute the index of this layer in the CloudVolume archive
        offset = cv.mip_voxel_offset(m)
        step = np.power(np.array(factor), m)
        cv_z_start = int(z_start // step[2] + offset[2])
        cv_z_end = int(min(cv_z_start + layer.shape[-1], cv.shape[-2]))

        # Set the layer
        cv[:, :, cv_z_start:cv_z_end] = layer

        # Reduce the size of the layer to match the next mip level
        layer = layer[::factor[0], ::factor[1], ::factor[2]]


def img2cv(input, output, mode='image', ext='.tif', resolution=(10, 10, 10),
           mip=0, chunk_size=(64, 64, 64), z_step=None, factor=(2, 2, 2),
           flip_xy=False, memory_limit=10000, offset=(0, 0, 0), quiet=False):
    """Create and write data to a new CloudVolume archive."""
    if quiet:
        LOGGER.logger.removeHandler(syslog)
        noop = logging.NullHandler()
        LOGGER.logger.addHandler(noop)

    outpath = os.path.abspath(output)

    if os.path.isdir(outpath) and mode not in os.path.basename(outpath):
        outpath = outpath +  '/{}'.format(mode)

    if not re.search(r'^file://.+$', outpath):
        outpath = 'file://' + outpath

    # On rank 0, initialize the CloudVolume info file, and load in the list of
    # images to insert into the archive.
    if RANK == 0:
        imagelist = sorted(glob.glob(os.path.join(input, '*' + ext)))
        img = io.imread(imagelist[0])
        dtype = img.dtype
        volume_shape = (len(imagelist),) + img.shape[:2]
        del img
        LOGGER.info('Converting {} with shape {} to CloudVolume'.format(input, volume_shape))
        LOGGER.info('Initialized CloudVolume image layer at {}'.format(outpath))
        initialize_cloudvolume(
            outpath,
            mode,
            dtype,
            resolution,
            offset,
            volume_shape[::-1],
            chunk_size,
            mip,
            factor)
        LOGGER.info('Broadcasting image list {}.'.format(imagelist))
    else:
        imagelist = None
        volume_shape = None

    # Send the CloudVolume parameters and list of images to all MPI ranks.
    imagelist = COMM.bcast(imagelist, root=0)
    volume_shape = COMM.bcast(volume_shape, root=0)

    # Iterate over layers of the volume. Each rank will load and write one
    # layer at a time. If there are fewer ranks than layers, increment to
    # n_ranks + rank and load the layer at that index.
    layer_idx = int(RANK * chunk_size[-1])
    while layer_idx < volume_shape[0]:
        # Compute the index of the first image in this layer, including any
        # offset from the volume origin.
        layer_shape = int(min(layer_idx + chunk_size[-1], len(imagelist)))
        LOGGER.info('Loading images {}-{}.'.format(layer_idx, layer_idx + layer_shape))
        layer = load_layer(imagelist, layer_idx, layer_shape)

        if mode == 'segmentation':
            layer = layer.astype(np.uint32)

        # Write the layer to the archive.
        LOGGER.info('Writing images {}-{}'.format(layer_idx, layer_idx + layer_shape))
        write_layer(
            outpath,
            mode,
            layer,
            flip_xy,
            layer_idx,
            mip,
            factor)

        # Increment to the next known layer that does not overlap with any
        # other rank.
        layer_idx += int(SIZE * chunk_size[-1])
    LOGGER.info('Done')


if __name__ == '__main__':
    args = parse_args()
    img2cv(args.input,
           args.output,
           mode=args.mode,
           ext=args.ext,
           resolution=args.resolution,
           mip=args.mip,
           chunk_size=args.chunk_size,
           z_step=args.z_step,
           factor=args.factor,
           flip_xy=args.flip_xy,
           memory_limit=args.memory_limit,
           offset=args.offset,
           quiet=args.quiet)

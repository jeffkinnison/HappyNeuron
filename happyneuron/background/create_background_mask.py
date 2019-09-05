#!/usr/bin/env python3
"""Compute a background mask for X-ray microscopy data.

Functions
---------
parse_args
    Parse command line arguments.
initialize_cloudvolume
    Create a new CloudVolume archive.
load_image
    Load an image from CloudVolume.
create_bg_mask
    Create a mask of background regions in x-ray microscopy.
write_image
    Write an image to CloudVolume.

Dependencies
------------
cloud-volume
mpi4py
numpy
scipy
scikit-image
"""

import argparse
import logging
import os
import re

from cloudvolume import CloudVolume
from mpi4py import MPI
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
from skimage.exposure import histogram
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

LOGGER = logging.getLogger('create_background_mask.py')
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
                   help='path to the CloudVolume archive')
    p.add_argument('--output', type=str,
                   help='path to the bg mask CloudVolume archive')
    p.add_argument('--resolution', type=int, nargs='*', default=[10, 10, 10],
                   help='resolution of the dataset')
    p.add_argument('--mip', type=int, default=0,
                   help='number of mip levels to create')
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
                   help='pass to deactivate logging')

    return p.parse_args()


def initialize_cloudvolume(path, resolution, offset, volume_size, chunk_size,
                           mip, factor):
    """Create a new CloudVolume archive.

    Parameters
    ----------
    path : str
        Filepath to the location to write the archive.
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
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type='segmentation',
        data_type='uint32',
        encoding='compressed_segmentation',
        resolution=resolution,
        voxel_offset=offset,
        volume_size=volume_size[:-1],
        chunk_size=chunk_size,
        max_mip=0,
        factor=factor
    )

    # Set up and initialize the CloudVolume object
    cv_args = dict(
        bounded=True, fill_missing=True, autocrop=False,
        cache=False, compress_cache=None, cdn_cache=False,
        progress=False, info=info, provenance=None, compress=True,
        non_aligned_writes=True, parallel=1)

    # for i in range(1, mip + 1):
    #     info['scales'][i]['compressed_segmentation_block_size'] = \
    #         info['scales'][0]['compressed_segmentation_block_size']

    cv = CloudVolume(path, mip=0, **cv_args)

    # Create the info file.
    LOGGER.info('Initializing image layer with config {}'.format(cv_args))
    cv.commit_info()
    return cv_args


def load_subvolume(cv, z_start, z_end, flip_xy=False):
    """Load an image from CloudVolume.

    Parameters
    ----------
    cv : cloudvolume.CloudVolume
        CloudVolume image layer to mask.
    z_start : int
        The index of the first image in the layer.
    z_end : int
        The index of the last image in the layer.
    flip_xy : bool
        CloudVolume reorders the dimension of image volumes, and the order of
        the x and y dimensions can vary. If True, indicates that the CloudVolume
        layer is saved in (Y, X, Z) order; otherwise it is saved as (X, Y, Z).

    Returns
    -------
    subvol : numpy.ndarray
        The subvolume with the dimensions reordered as (Z, Y, X).
    """
    # Each entry in the z dimension represents one image. Extract an image.
    subvol = cv[:, :, z_start:z_end, :]
    subvol = np.squeeze(subvol)

    # Transpose the dimensions back to
    if not flip_xy:
        subvol = np.transpose(subvol, axes=[2, 1, 0])

    LOGGER.info('Loaded subvolume with shape {}.'.format(subvol.shape))
    return subvol


def find_bg_mask(img):
    """Create a mask of background regions in x-ray microscopy.

    Parameters
    ----------
    img : numpy.ndarray
        X-ray microscopy image.

    Returns
    -------
    bgmask : numpy.ndarray
        Binary mask of the background of ``img``.
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)

    bgmask = np.zeros((3,) +img.shape, dtype=np.uint8)

    for d in range(img.ndim):
        for i in range(img.shape[d]):
            if d == 0:
                subimg = img[i, :, :]
            elif d == 1:
                subimg = img[:, i, :]
            elif d == 2:
                subimg = img[:, :, i]

            # Blur the image to smooth any background artifacts.
            LOGGER.info('Blurring image.')
            blur = gaussian(subimg, sigma=5, preserve_range=True)

            # Compute the image histogram and find the peaks.
            LOGGER.info('Finding histogram peaks.')
            hist, bins = histogram(blur)
            peaks, properties = find_peaks(hist)  # , height=(0.3 * img.size))
            prominences = peak_prominences(hist, peaks)
            widths = peak_widths(hist, peaks, rel_height=0.333,
                                 prominence_data=prominences)

            # Select the left-most peak (backgrounds are usually dark) and use the
            # width of the peak to select a threshold value. Create a mask of all
            # pixels less than or equal to the threshold.
            ordered = np.argsort(peaks)
            threshold = peaks[ordered[0]] + (widths[0][ordered[0]] / 2.0)
            # threshold = peaks[0] + (widths[0][0] / 2.0)
            LOGGER.info('Setting hard threshold {} for image.'.format(threshold))
            mask = np.zeros(subimg.shape, dtype=np.uint8)
            mask[np.where(subimg <= threshold)] = 1
            # Perform some clean up and find the largest connected component.
            LOGGER.info('Cleaning mask of image.')
            # remove_small_holes(mask, area_threshold=30, connectivity=2,
            #                    in_place=True)
            labels = label(mask)
            objs = regionprops(labels)
            # bg = None
            # for obj in objs:
            #     if obj.bbox_area >= 0.85 * img.size:
            #         coords = obj.coords
            #         break

            # Select the connected component with the largest bounding box as the
            # background mask.
            objs.sort(key=lambda x: x.bbox_area, reverse=True)
            # objs = [o for o in objs
            #         if np.any(np.asarray(o.bbox[:mask.ndim]) == np.asarray(mask.shape))
            #         or np.any(np.asarray(o.bbox[mask.ndim:]) == 0)]
            print(len(objs))
            if len(objs) > 0:
                coords = tuple([objs[0].coords[:, j] for j in range(subimg.ndim)])
                LOGGER.info('Setting background mask of image.')

                if d == 0:
                    bgmask[d, i, coords[0], coords[1]] = 1
                elif d == 1:
                    bgmask[d, coords[0], i, coords[1]] = 1
                elif d == 2:
                    bgmask[d, coords[0], coords[1], i] = 1
    LOGGER.info('Full background mask covers {} voxels.'.format(np.sum(bgmask)))

    consensus = bgmask[0] + bgmask[1] + bgmask[2]
    consensus[np.where(consensus == 1)] = 0
    consensus[consensus.nonzero()] = 1
    objs = sorted(regionprops(label(consensus)), key=lambda x: x.bbox_area, reverse=True)
    for obj in objs[1:]:
        coords = tuple([obj.coords[:, j] for j in range(img.ndim)])
        consensus[coords] = 0

    LOGGER.info('Full background mask covers {} voxels.'.format(np.sum(consensus)))
    return consensus.astype(np.uint32)


def write_subvolume(path, subvolume, flip_xy, z_start, mip, factor):
    """Write an image to CloudVolume.

    Parameter
    ---------
    path : str
        Filepath to the location to write the archive.
    subvolume : numpy.ndarray
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
    """
    # Transpose the axes to match the CloudVolume order
    if subvolume.ndim == 2:
        subvolume = np.expand_dims(subvolume, 0)

    if flip_xy:
        subvolume = np.transpose(subvolume, axes=[1, 2, 0])
    else:
        subvolume = np.transpose(subvolume, axes=[2, 1, 0])

    if subvolume.ndim == 3:
        subvolume = np.expand_dims(subvolume, -1)

    cv_args = dict(
        bounded=True, fill_missing=True, autocrop=False,
        cache=False, compress_cache=None, cdn_cache=False,
        progress=False, info=None, provenance=None, compress=True,
        non_aligned_writes=True, parallel=1)

    # Set the volume for each mip level
    for m in range(1):
        # Access the CloudVolume
        LOGGER.info('Writing MIP level {}.'.format(mip))
        cv = CloudVolume(path, mip=m, **cv_args)

        # Compute the index of this layer in the CloudVolume archive
        offset = cv.mip_voxel_offset(m)
        step = np.power(np.array(factor), m)
        cv_z_start = int(z_start // step[2] + offset[2])
        cv_z_end = int(min(cv_z_start + subvolume.shape[-2], cv.shape[-2]))

        # Set the layer
        cv[:, :, cv_z_start:cv_z_end] = subvolume

        # Reduce the size of the layer to match the next mip level
        subvolume = subvolume[::factor[0], ::factor[1], ::factor[2]]


def create_background_mask(input, output, resolution=(10, 10, 10), mip=0,
                           chunk_size=(64, 64, 64), z_step=None,
                           factor=(2, 2, 2), flip_xy=False, memory_limit=10000,
                           offset=(0, 0, 0), quiet=False):
    """Create and write data to a new CloudVolume archive."""
    if quiet:
        LOGGER.logger.removeHandler(syslog)
        noop = logging.NullHandler()
        LOGGER.logger.addHandler(noop)

    if 'image' not in os.path.basename(input):
        inpath = input + '/image'
    else:
        inpath = input

    if os.path.isdir(inpath) and not re.search(r'^file://', inpath):
        inpath = 'file://' + os.path.abspath(inpath)

    if RANK == 0:
        LOGGER.info('Loading CloudVolume image layer {}.'.format(inpath))

    img_cv = CloudVolume(inpath)
    volume_shape = img_cv.shape

    outpath = os.path.abspath(output)

    if os.path.dirname(inpath) == outpath:
        outpath = outpath + 'background'

    if not re.search(r'^[\w]+://.+$', outpath):
        outpath = 'file://' + os.path.abspath(output)

    # On rank 0, initialize the CloudVolume info file, and load in the list of
    # images to insert into the archive.
    if RANK == 0:
        LOGGER.info('Initialized CloudVolume image layer at {}'.format(outpath))
        cv_args = initialize_cloudvolume(
            outpath,
            resolution,
            offset,
            volume_shape,
            chunk_size,
            mip,
            factor)

    # Block until the background CloudVolume layer is initialized.
    GOGOGO = COMM.bcast(1, root=0)

    # Iterate over layers of the volume. Each rank will load and write one
    # layer at a time. If there are fewer ranks than layers, increment to
    # n_ranks + rank and load the layer at that index.
    # offset from the volume origin.
    layer_idx = RANK * chunk_size[-1]
    while layer_idx < volume_shape[-2]:
        # Compute the index of the first image in this layer, including any
        layer_shape = int(min(layer_idx + chunk_size[-1],
                              img_cv.shape[-2]))
        LOGGER.info('Loading images {}-{}.'.format(layer_idx, layer_shape))
        image = load_subvolume(img_cv, layer_idx, layer_shape,
                               flip_xy=flip_xy)

        LOGGER.info('Creating background mask.')
        mask = find_bg_mask(image)

        # Write the layer to the archive.
        LOGGER.info('Writing mask of images {}-{}'.format(layer_idx, layer_shape))
        write_subvolume(
            outpath,
            mask,
            flip_xy,
            layer_idx,
            mip,
            factor)

        # Increment to the next known layer that does not overlap with any
        # other rank.
        layer_idx += SIZE * chunk_size[-1]

    LOGGER.info('Done.')

if __name__ == '__main__':
    args = parse_args()
    create_background_mask(
        args.input,
        args.output,
        resolution=args.resolution,
        mip=args.mip,
        chunk_size=args.chunk_size,
        z_step=args.z_step,
        factor=args.factor,
        flip_xy=args.flip_xy,
        memory_limit=args.memory_limit,
        offset=args.offset,
        quiet=args.quiet)

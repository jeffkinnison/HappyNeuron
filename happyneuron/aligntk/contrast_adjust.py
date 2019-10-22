import numpy as np
from scipy.misc import ascent
from matplotlib import pyplot as plt
import argparse
import os
import glob
import cv2
from pprint import pprint
from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()


def hist_norm(x, bin_edges, quantiles, inplace=False):
    """
    Adapted from:
    https://stackoverflow.com/questions/31490167/how-can-i-transform-the-histograms-of-grayscale-images-to-enforce-a-particular-r?rq=1

    Linearly transforms the histogram of an image such that the pixel values
    specified in `bin_edges` are mapped to the corresponding set of `quantiles`

    Arguments:
    -----------
        x: np.ndarray
            Input image; the histogram is computed over the flattened array
        bin_edges: array-like
            Pixel values; must be monotonically increasing
        quantiles: array-like
            Corresponding quantiles between 0 and 1. Must have same length as
            bin_edges, and must be monotonically increasing
        inplace: bool
            If True, x is modified in place (faster/more memory-efficient)

    Returns:
    -----------
        x_normed: np.ndarray
            The normalized array
    """

    bin_edges = np.atleast_1d(bin_edges)
    quantiles = np.atleast_1d(quantiles)

    if bin_edges.shape[0] != quantiles.shape[0]:
        raise ValueError('# bin edges does not match number of quantiles')

    if not inplace:
        x = x.copy()
    oldshape = x.shape
    pix = x.ravel()

    # get the set of unique pixel values, the corresponding indices for each
    # unique value, and the counts for each unique value
    pix_vals, bin_idx, counts = np.unique(pix, return_inverse=True,
                                          return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution function (which maps pixel
    # values to quantiles)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]

    # get the current pixel value corresponding to each quantile
    curr_edges = pix_vals[ecdf.searchsorted(quantiles)]

    # how much do we need to add/subtract to map the current values to the
    # desired values for each quantile?
    diff = bin_edges - curr_edges

    diff[0] = 0
    diff[1] = 0
    diff[-1] = 0
    # interpolate linearly across the bin edges to get the delta for each pixel
    # value within each bin
    pix_delta = np.interp(pix_vals, curr_edges, diff)

    # add these deltas to the corresponding pixel values
    #print(pix.dtype, pix_delta.dtype)
    pix += np.floor(pix_delta[bin_idx]).astype(np.uint8)

    return pix.reshape(oldshape)

def hist_norm_w_mask(x, mask, bin_edges, quantiles, inplace=False):
    bin_edges = np.atleast_1d(bin_edges)
    quantiles = np.atleast_1d(quantiles)

    if bin_edges.shape[0] != quantiles.shape[0]:
        raise ValueError('# bin edges does not match number of quantiles')

    if not inplace:
        x = x.copy()
    oldshape = x.shape
    pix = x.ravel()
    mask1d = (mask>0).ravel()

    pix_w_mask = np.copy(pix)
    pix_w_mask[mask1d] = 0

    pix_vals, bin_idx, counts = np.unique(pix_w_mask, return_inverse=True,
                                          return_counts=True)
    masked_counts = counts
    masked_counts[0] = 0

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution function (which maps pixel
    # values to quantiles)
    ecdf = np.cumsum(masked_counts).astype(np.float64)
    ecdf /= ecdf[-1]

    # get the current pixel value corresponding to each quantile
    curr_edges = pix_vals[ecdf.searchsorted(quantiles)]

    # how much do we need to add/subtract to map the current values to the
    # desired values for each quantile?
    diff = bin_edges - curr_edges

    # interpolate linearly across the bin edges to get the delta for each pixel
    # value within each bin

    diff[0] = 0
    diff[1] = 0
    diff[-1] = 0
    # print('pix_vals:', pix_vals)
    # print('curr_edges:', curr_edges)
    # print('diff:', diff)
    pix_delta = np.interp(pix_vals, curr_edges, diff)

    # add these deltas to the corresponding pixel values
    pix += np.floor(pix_delta[bin_idx]).astype(np.uint8)

    return pix.reshape(oldshape)
    

def ecdf(x):
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


def mpi_worker(input_list, mask_dir, output_dir, bin_edges, quantiles):
  for fi in tqdm(input_list):
    bname = os.path.basename(fi).split('.')[0]
    fo = os.path.join(output_dir, bname+'.tif')
    img = cv2.imread(fi, 0)
    if mask_dir:
        fm = os.path.join(mask_dir, bname+'.pbm')
        mask = cv2.imread(fm, 0)
        img = hist_norm_w_mask(img, mask, bin_edges, quantiles)
    else:
        img = hist_norm(img, bin_edges, quantiles)
    cv2.imwrite(fo, img)

def mpi_run(input_dir, mask_dir, output_dir, f_ref_image, f_ref_mask, delta):
  if mpi_rank == 0:
    f_list = glob.glob(os.path.join(input_dir, '*.tif*'))
    f_list.sort()
    f_sublist = np.array_split(np.asarray(f_list), mpi_size)

    os.makedirs(output_dir, exist_ok=True)

    ref = cv2.imread(f_ref_image, 0)
    if f_ref_mask:
        ref_mask = cv2.imread(f_ref_mask, 0)
        ref1d = ref[ref_mask==0].ravel()
        x1, y1 = ecdf(ref1d)
    else:
        ref1d = ref.ravel()
        x1, y1 = ecdf(ref1d)

    bin_edges = np.arange(0,255,delta)
    quantiles = []
    for b in bin_edges:
      q = np.where(x1>b)[0]
      if len(q) > 1:
        quantiles.append(y1[q[0]])
      else:
        quantiles.append(1.0)
    # quantiles = [y1[np.where(x1>b)[0][0]] for b in bin_edges[:]]

  else:
    f_sublist = None
    bin_edges = None
    quantiles = None

  f_sublist = mpi_comm.scatter(f_sublist, root=0)
  bin_edges = mpi_comm.bcast(bin_edges, root=0)
  quantiles = mpi_comm.bcast(quantiles, root=0)
  
  mpi_worker(f_sublist, mask_dir, output_dir, bin_edges, quantiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--mask', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--ref_image', type=str, default=None)
    parser.add_argument('--ref_mask', type=str, default=None)
    parser.add_argument('--delta', type=int, default=10)

    args = parser.parse_args()

    mpi_run(args.input, args.mask, args.output, args.ref_image, args.ref_mask, args.delta)


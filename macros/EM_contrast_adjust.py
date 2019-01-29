import numpy as np
from scipy.misc import ascent
from matplotlib import pyplot as plt
from .reader import *
import dxchange
import argparse
import os
from joblib import Parallel, delayed
from tqdm import tqdm
def hist_norm(x, bin_edges, quantiles, inplace=False):
    """
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

    # interpolate linearly across the bin edges to get the delta for each pixel
    # value within each bin
    pix_delta = np.interp(pix_vals, curr_edges, diff)

    # add these deltas to the corresponding pixel values
    #print(pix.dtype, pix_delta.dtype)
    pix += np.floor(pix_delta[bin_idx]).astype(np.uint8)

    return pix.reshape(oldshape)

def ecdf(x):
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf
def demo():
    bin_edges = 0, 55, 200, 255
    quantiles = 0, 0.2, 0.5, 1.0
    img = ascent()
    normed = hist_norm(img, bin_edges, quantiles)


    x1, y1 = ecdf(img.ravel())
    x2, y2 = ecdf(normed.ravel())

    fig = plt.figure()
    gs = plt.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[1, :])
    for aa in (ax1, ax2):
        aa.set_axis_off()

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax2.imshow(normed, cmap=plt.cm.gray)
    ax2.set_title('Normalised')

    ax3.plot(x1, y1 * 100, lw=2, label='Original')
    ax3.plot(x2, y2 * 100, lw=2, label='Normalised')
    for xx in bin_edges:
        ax3.axvline(xx, ls='--', c='k')
    for yy in quantiles:
        ax3.axhline(yy * 100., ls='--', c='k')
    ax3.set_xlim(bin_edges[0], bin_edges[-1])
    ax3.set_xlabel('Pixel value')
    ax3.set_ylabel('Cumulative %')
    ax3.legend(loc=2)
    plt.show()

def run(f_input, f_output, stack_begin, stack_end, z_step):
    ''' f_input: first slice of a tiff stack
        f_output: output dir
    '''
    data_generator = omni_generator(f_input, begin=stack_begin, end=stack_end, step=z_step)
    ref = omni_read(f_input, begin=stack_begin, end=stack_begin+1)
    x1, y1 = ecdf(ref)

    #bin_edges = [0, 50, 100, 150, 200, 255]
    #quantiles = [y1[e] for e in bin_edges]

    quantiles = np.arange(0.1,1,0.1)
    bin_edges = [np.where(y1>q)[0][0] for q in quantiles[:]]

    print(bin_edges)
    print(quantiles)

    for (i, data) in tqdm(data_generator, total=(stack_end-stack_begin)//z_step):
        #print(i,data.shape)
        L = data.shape[0]
        data_out = Parallel(n_jobs=L)(
            delayed(hist_norm)(data[i], bin_edges, quantiles, inplace=False)
            for i in range(L)
        )
        dxchange.write_tiff_stack(np.stack(data_out), os.path.join(f_output, 'data'), start=i, overwrite=True)
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--begin', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--step', type=int, default=1)

    args = parser.parse_args()

    if not args.begin and not args.end:
        S,L = check_stack_len(args.input)
        BEGIN = S
        END = S+L
    else:
        BEGIN = args.begin
        END = args.end

    run(args.input, args.output,BEGIN,END,args.step)

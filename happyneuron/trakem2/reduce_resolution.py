"""Reduce Resolution of EM images."""

import argparse
import glob
import os

import cv2
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, type=str,
                        help='path to images to downsample')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='path to write downsampled images')
    parser.add_argument('--factor', default=2, type=int,
                        help='factor to downsample by')
    args = parser.parse_args()


def reduce_resolution(input_dir, output_dir, factor=2):
    """Reduce the resolution of EM images.

    Parameters
    ----------
    input_dir : str
        Path to images to downsample.
    output_dir : str
        Path to write downsampled images.
    factor : int
        Factor to downsample by. Default: 2.
    """
    # Get the images.
    downsample_ratio = 1.0 / args.factor
    tiles = glob.glob(os.path.join(args.input_dir, 'S_*/*.tif*'))
    tiles.sort()
    print('Number of Tiles: ', len(tiles))
    print('Downsample Ratio: ', downsample_ratio)

    # Downsample the images by the requested factor with OpenCV's resize().
    for t in tqdm(tiles):
        dirname = os.path.basename(os.path.dirname(t))
        filename = os.path.basename(t)
        os.makedirs(os.path.join(args.output_dir, dirname), exist_ok=True)
        f_out = os.path.join(args.output_dir, dirname, filename)
        im = cv2.imread(t, 0)
        im_down = cv2.resize(im, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
        cv2.imwrite(f_out, im_down)


def main():
    args = parse_args()
    reduce_resolution(args.input_dir, args.output_dir, factor=args.factor)

if __name__ == '__main__':
    main()

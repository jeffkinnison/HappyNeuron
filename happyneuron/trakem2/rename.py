"""Rename TrakEM2 output to match *[index].tif."""

import argparse
import glob
import os
import re
import sys

import h5py


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        help='path to the input images to convert')
    parser.add_argument('--output_dir', type=str, default=None
                        help='path to write converted files to')
    return parser.parse_args(args)


def rename(input_dir, output_dir=None):
    """Rename TrakEM2 output to match *[index].tif.

    TrakEM2 outputs image files with a particular filename pattern. This
    function will strip every element except the image index from the
    filenames.

    Parameters
    ----------
    input_dir : str
        Path to the input images to convert.
    output_dir : str, optional
        Path to write converted files to. If not provided, input images are
        overwritten.
    """
    def get_index(f):
        return int(re.match(
            r'^.*/.*_z(?P<index>[\d]+).(.*)$', f).groupdict('index')['index'])

    print(f_in_dir)
    f_list = glob.glob(os.path.join(f_in_dir, '*.tif*'))

    for f in f_list:
        new_name = os.path.join(
            f_in_dir, 'S_' + str.zfill(str(get_index(f)), 4) + '.tif')
        print(new_name)
        os.rename(f, new_name)


def main():
    args = parse_args()
    print(args.input)
    rename(args.input_dir, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

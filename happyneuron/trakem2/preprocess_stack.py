"""Create a trakem2 input text file from a directory of TIFF stacks.

Preprocess a directory with S_0000.tif, S_0001.tif, ... S_*.tif into trakem2 import text file
"""

from __future__ import print_function, division

import argparse
from ast import literal_eval as make_tuple
import glob
import os
from pprint import pprint
import re
import shutil
import sys

import cv2
from matplotlib.pyplot import *
import numpy as np
from PIL import Image
import tifffile
from tqdm import tqdm


class EMStackPreprocessor(object):
    """Set up EM TIFF stacks for alignment.

    Parameters
    ----------
    input_dir : str
        Path to the input images.
    output : str
        Output .txt file with

    Attributes
    ----------
    flist : str
        List of input image files. Populated by ``EMStackPreprocessor.run``.
    MAX_ROW : int
    MAX_COL : int
        Unused.
    TILE_ROW : int
    TILE_COL : int
        The number of rows and columns in a given tile within the volume.
    TILE_MIN : int
    TILE_MAX : int
        The min and max grayscale values in all tiles.
    DTYPE : int
        Data type of the images to align.
    """

    def __init__(self, input_dir, output):
        self.input_dir = input_dir
        self.output = output
        output_dir = os.path.dirname(self.output)
        os.makedirs(output_dir, exist_ok=True)

        self.flist = None
        self.MAX_ROW = 0
        self.MAX_COL = 0

        self.TILE_ROW = 0
        self.TILE_COL = 0
        self.TILE_MIN = 0
        self.TILE_MAX = 0
        self.DTYPE = 0

    def test_one_image(self):
        """Get metadata from an input image."""
        # Select one image from the set and open it.
        f_dummy = glob.glob(os.path.join(self.input_dir, '*.*'))[0]
        dummy_data = cv2.imread(f_dummy, flags=cv2.IMREAD_GRAYSCALE)
        print(dummy_data.shape)

        # Extract shape and grayscale information.
        self.TILE_ROW, self.TILE_COL = dummy_data.shape
        # This line extracts the min and max grayscale values for the dummy
        # data, but not for the entire volume. Should we be extracting for the
        # entire volume or setting it to the data type min/max values instead?
        self.TILE_MIN, self.TILE_MAX = np.min(dummy_data[:]), np.max(dummy_data[:])
        print(self.TILE_ROW, self.TILE_COL, self.TILE_MIN,
              self.TILE_MAX, dummy_data.dtype)

        # Extract image data type information.
        if dummy_data.dtype == np.uint8:
            print('8bit')
            self.DTYPE = 0
        elif dummy_data.dtype == np.uint16:
            print('16bit')
            self.DTYPE = 1

    def prepare_align_txt(self):
        """Write out the trakem2 input file."""
        flist = np.asarray(glob.glob(os.path.abspath(
            os.path.join(self.input_dir, '*.*'))))

        # Sort the files by their index.
        inds = [int(re.search('.*_([0-9]*)', f.split('/')[-1]).group(1))
                for f in flist]
        flist = flist[np.argsort(inds)]

        # Write the command for each image to the aligntk input file.
        with open(self.output, 'w') as output:
            for i, f in enumerate(flist):
                command = '{0} \t {1} \t {2} \t {3} \t {4} \t {5} \t {6} \t {7} \t {8} \n'.format(
                    f, 0, 0, i, self.TILE_COL, self.TILE_ROW, self.TILE_MIN, self.TILE_MAX, self.DTYPE)
                print(command)
                output.write(command)

    def run(self):
        """Preprocess image data to run through aligntk."""
        print("Input:", self.input_dir)
        print("Output:", self.output)

        # Get the TIFF stack filenames.
        self.flist = glob.glob(os.path.join(self.input_dir, 'S_*'))

        pprint(self.flist)

        # Get image metadata.
        self.test_one_image()

        # Write out the aligntk input text file.
        self.prepare_align_txt()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='')
    parser.add_argument('output', type=str, help='')
    return parser.parse_args(args)


def main():
    args = parse_args()
    emp = EMStackPreprocessor(args.input, args.output)
    emp.run()


if __name__ == '__main__':
  main()

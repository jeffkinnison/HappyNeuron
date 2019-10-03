from __future__ import print_function, division

import argparse
from ast import literal_eval as make_tuple
import glob
import os
from pprint import pprint
import re
import shutil
import sys
import tifffile

import cv2
from matplotlib.pyplot import *
import numpy as np
from PIL import Image
from tqdm import tqdm


class EMPrealignPreprocessor(object):
    """Set up EM TIFF stacks for alignment.

    Parameters
    ----------
    input_dir : str
        Path to the input images.
    output : str
        Output .txt file with
    trakem2_index : bool
        If true, assume filenames follow trakem2 naming convention.

    Attributes
    ----------
    flist : str
        List of input image files.
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

    def __init__(self, input_dir, output_dir, trakem2_index):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)

        if output_dir == None:
            output_dir = os.path.join(self.input_dir, 'output')
        try:
            os.makedirs(output_dir)
        except:
            pass
        self.output_dir = output_dir
        self.flist = glob.glob(os.path.join(self.input_dir, '*.tif*'))
        if trakem2_index:
            def get_index(f): return int(
                re.match(r'^.*/.*_z(?P<index>[0-9]+).(.*)$', f).groupdict('index')['index'])
            self.flist.sort(key=get_index)
        else:
            def get_index(f): return int(
                re.match(r'^.*/.*_(?P<index>[0-9]+)\.(.*)$', f).groupdict('index')['index'])
            self.flist.sort(key=get_index)

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
        print(len(self.flist))
        f_dummy = self.flist[0]
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
        f_align_txt = os.path.join(self.output_dir, 'align.txt')
        with open(f_align_txt, 'w') as output:
            for i, f in enumerate(self.flist):
                command = '{}\t{}\t{}\t{}\n'.format(f, '0', '0', str(i))
                output.write(command)

    def run(self):
        """Preprocess image data to run through aligntk."""
        print("Input:", self.input_dir)
        print("Output:", self.output_dir)

        # Step 5: prepare TrackEM2 import txt file
        #self.prepare_align_txt(self.input_dir, self.input_dir)
        self.test_one_image()
        self.prepare_align_txt()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='dir containing the stack, with *[id].tif')
    parser.add_argument('--output', default='.')
    parser.add_argument('--trakem2_index', action='store_true',
                        help='whether the tiff stack uses trackem2 style "*z[id].0.tif" indexing ')
    args = parser.parse_args(args)


def main():
    args = parse_args()
    emp = EMPrealignPreprocessor(
        args.input, args.output, args.trackem2_index)
    emp.run()


if __name__ == '__main__':
    main()

"""Set up EM images for alignment.

Classes
-------
EMPreprocessor
    Set up EM images for alignment.
"""

from __future__ import print_function, division

import argparse
import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np
from tqdm import tqdm


class EMTilePreprocessor(object):
    """Set up EM images for alignment.

    Parameters
    ----------
    input_dir : str
        Path to the input images.
    output : str
        Output .txt file with

    Attributes
    ----------
    flist : str
        List of input image files. Populated by ``EMTilePreprocessor.run``.
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

    def __init__(self, input_dir, output='align.txt'):
        self.input_dir = input_dir
        self.output = os.path.abspath(output)
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
        f_dummy = glob.glob(os.path.join(self.input_dir, 'S_*/Tile*.tif'))[0]
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
        with open(self.output, 'w') as f_out:
            for f in self.flist:
                tlist = glob.glob(os.path.join(f, 'Tile_*.tif'))

                # This seems superfluous--can it be removed?
                if len(tlist) == 0:
                    continue

                # For every image file, get the filename and its coordinates in
                # the entire imaged volume. Then, write out the trakem2 command
                # to file, with additional metadata.
                for t in tlist:
                    res = re.search(r'Tile_r([0-9])-c([0-9])_S_([0-9]+)_*', t)
                    tile_name = os.path.abspath(t)
                    r = res.group(1)
                    c = res.group(2)
                    z = int(res.group(3))

                    command = '{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n'.format(
                      tile_name, c, r, z, self.TILE_COL, self.TILE_ROW, self.TILE_MIN, self.TILE_MAX, self.DTYPE)
                    f_out.write(command)

    def run(self):
        """Preprocess image data to run through trakem2."""
        print("Input:", self.input_dir)
        print("Output:", self.output)

        # Get the image filenames and sort by their index.
        self.flist = glob.glob(os.path.join(self.input_dir, 'S_*'))

        def get_index(f):
            """Retrieve the index from an image filename.

            Returns
            -------
            idx : str
                The string index of the image file.
            """
            return re.search(r'([0-9]+)', os.path.basename(f)).group(1)

        self.flist.sort(key=get_index)

        # Get image metadata.
        self.test_one_image()

        # Write out the trakem2 input text file.
        self.prepare_align_txt()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='Directory of a section_set/site')
    parser.add_argument('output', type=str, default='align.txt',
                        help='Output .txt file')
    return parser.parse_args(args)


def main():
    args = parse_args()
    emp = EMTilePreprocessor(args.input, args.output)
    emp.run()


if __name__ == '__main__':
    main()

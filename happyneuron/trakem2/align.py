"""Run trakem alignment in parallel with MPI."""

import os
import argparse
import numpy as np
import cv2
import re
import glob
from PIL import Image
from tqdm import tqdm
import logging
import pkg_resources


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default=None, type=str,
                        help='path to trakem2 input file')
    parser.add_argument('output', default=None, type=str,
                        help='path to output aligned images')
    parser.add_argument('--pairs', default=None, type=str,
                        help='path to pairs definition file')
    parser.add_argument('--min', default=1024, type=int,
                        help='minimum image size for SIFT feature extraction')
    parser.add_argument('--max', default=2048, type=int,
                        help='maximum image size for SIFT feature extraction')
    parser.add_argument('--begin', default=0, type=int,
                        help='the index of the first image to align')
    parser.add_argument('--fiji', default='fiji', type=str,
                        help='specify ImageJ-linux64 executable path')
    return parser.parse_args(args)


def align(input, output, pairs=None, min_octave=1024, max_octave=2048,
          begin=0, fiji='fiji'):
    """Run trakem2 alignment.

    Parameters
    ----------
    input : str
        Path to the trakem2 input file.
    output : str
        Path to write subdivided trakem2 inputs.
    pairs : str
        Path to the pairs definition file.
    min_octave : int
        Minimum image size for SIFT feature extraction. Default: 1024.
    max_octave : int
        Maximum image size for SIFT feature extraction. Default: 2048
    begin : int
        The index of the first image to align.
    fiji : str
        Path to an ImageJ-linux64 executable.
    """
    # Set up the output directory and get the ImageJ script to run.
    os.makedirs(args.output, exist_ok=True)
    resource_path = 'ext/align.bsh'
    bsh_path = pkg_resources.resource_filename(__name__, resource_path)
    logging.warning('macro: %s', bsh_path)

    # Set up the align command.
    if args.pairs:
        command = '%s --headless -Dinput=%s -Doutput=%s -Dpairs=%s -Dmin=%d -Dmax=%d -Dbegin=%d -- --no-splash %s' % (
            args.fiji, args.input, args.output, args.pairs, args.min, args.max, args.begin, bsh_path)
    else:
        command = '%s --headless -Dinput=%s -Doutput=%s -Dmin=%d -Dmax=%d -Dbegin=%d -- --no-splash %s' % (
            args.fiji, args.input, args.output, args.min, args.max, args.begin, bsh_path)

    # Align the volume.
    print(command)
    os.system(command)


def main():
    args = parse_args()
    align(input,
          output,
          pairs=args.pairs,
          min_octave=args.min,
          max_octave=args.max,
          begin=args.begin,
          fiji=args.fiji)


if __name__ == '__main__':
    main()

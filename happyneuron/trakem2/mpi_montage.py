"""Run trakem image montaging in parallel with MPI."""

import argparse
import glob
import logging
import os
import re

import cv2
import numpy as np
from mpi4py import MPI
import pkg_resources
from PIL import Image
from tqdm import tqdm


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default=None, type=str,
                        help='path to align.txt')
    parser.add_argument('output', default=None, type=str,
                        help='path to subdivided trakem2 inputs')
    parser.add_argument('--min', default=1024, type=int,
                        help='minimum image size for SIFT feature extraction')
    parser.add_argument('--max', default=2048, type=int,
                        help='maximum image size for SIFT feature extraction')
    parser.add_argument('--fiji', default='fiji', type=str,
                        help='path to ImageJ-linux64 executable')
    return parser.parse_args(args)


def split_aligntxt(align_txt, output_dir):
    """Split an trakem2 input file into subsets for parallel processing.

    Parameters
    ----------
    align_txt : str
        Path to the trakem2 input file.
    output_dir : str
        Path to write subdivided trakem2 inputs.
    """
    # Read in trakem2 commands from the contiguous input file.
    with open(align_txt, 'r') as f:
        text = f.readlines()

    get_key = lambda x: int(re.search(r'S_(\d+)_', x).group(1))
    key_line_dict = {}

    # For each input, extract the z-index of the input and group files on the
    # same z-plane.
    for line in text:
        key = get_key(line)
        val = key_line_dict.get(key, [])
        val.append(line)
        key_line_dict[key] = val

    # Split the z-indices evenly among MPI ranks.
    key_set = np.asarray(list(key_line_dict.keys()))
    key_sublists = np.array_split(key_set, SIZE)

    # For each rank, write the assigned commands to an independent file.
    for i, keys in enumerate(key_sublists):
        with open(os.path.join(output_dir, 'align_%d.txt' % i), 'w') as f:
            for key in keys:
                f.writelines(key_line_dict[key])


def mpi_montage(input, output, min_octave=1024, max_octave=2048, fiji='fiji'):
    """Run trakem2 image montaging in parallel with MPI.

    Parameters
    ----------
    input : str
        Path to the trakem2 input file.
    output : str
        Path to write subdivided trakem2 inputs.
    min_octave : int
        Minimum image size for SIFT feature extraction. Default: 1024.
    max_octave : int
        Maximum image size for SIFT feature extraction. Default: 2048
    fiji : str
        Path to an ImageJ-linux64 executable.
    """
    # On rank 0, split the contents of the trakem2 input file evenly across
    # SIZE files and find the ImageJ script to run.
    if RANK == 0:
        os.makedirs(output, exist_ok=True)
        resource_path = 'ext/montage_macro.bsh'
        bsh_path = pkg_resources.resource_filename(__name__, resource_path)
        logging.warning('macro: %s', bsh_path)
        split_aligntxt(input, output)
    else:
        bsh_path = None

    # Sync ranks and give everyone the ImageJ script.
    bsh_path = COMM.bcast(bsh_path, 0)

    # Execute trakem2 image montaging on each rank in parallel.
    rank_input = os.path.join(output, 'align_%d.txt' % RANK)
    command = '%s --headless -Dinput=%s -Doutput=%s -Dmin=%d -Dmax=%d -- --no-splash %s' % (
      fiji, rank_input, output, min_octave, max_octave, bsh_path)
    print(command)
    os.system(command)


def main():
    args = parse_args()
    mpi_montage(args.input,
                args.output,
                min_octave=args.min,
                max_octave=args.max,
                fiji=args.fiji)


if __name__ == '__main__':
    main()

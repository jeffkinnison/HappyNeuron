"""
"""
import argparse
import glob
import logging
import re
import os

import cv2
from mpi4py import MPI
import numpy as np
from PIL import Image
import pkg_resources
from tqdm import tqdm


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default=None, type=str,
                        help='path to the trakem2 input file')
    parser.add_argument('output', default=None, type=str,
                        help='path to write output images')
    parser.add_argument('--range', default=None, type=int, nargs=2,
                        help='start and end of the image sequence to process')
    parser.add_argument('--fiji', default='fiji', type=str,
                        help='path to ImageJ-linux64 executable')
    return parser.parse_args(args)


def split_aligntxt(align_txt, output_dir):
    """Split a trakem2 input file evenly across MPI ranks.

    Parameters
    ----------
    align_txt : str
        Path to the trakem2 input file.
    output_dir : str
        Path to write subdivided trakem2 inputs.
    """
    # Read in trakem2 commands from the contiguous input file.
    with open(align_txt, 'r') as f:
        lines = f.readlines()

    get_key = lambda x: int(re.search(r'S_(\d+)_', x).group(1))
    key_line_dict = {get_key(l):l for l in lines}

    # For each input, extract the z-index of the input and group files on the
    # same z-plane.
    key_set = np.asarray(list(key_line_dict.keys()))
    key_sublists = np.array_split(key_set, SIZE)

    # For each rank, write the assigned commands to an independent file.
    for i, keys in enumerate(key_sublists):
        with open(os.path.join(output_dir, 'align_%d.txt' % i), 'w') as f:
            for key in keys:
                f.writelines(key_line_dict[key])


def get_keys(align_txt):
    """Get the z-indices of a list of images.

    Parameters
    ----------
    align_txt : str
        Path to the trakem2 input file.

    Return
    ------
    key_sublists : list of list of int
        List of image indices split evenly into SIZE groups.
    """
    # Get filenames to be exported.
    with open(align_txt, 'r') as f:
        lines = f.readlines()

    # Extract the image index.
    get_key = lambda x: int(re.search(r'S_(\d+).*', x).group(1))
    keys = np.asarray([get_key(l) for l in lines])
    keys.sort()

    # Split images evenly amongst MPI ranks.
    key_sublists = np.array_split(keys, SIZE)
    return key_sublists


def mpi_export(input, output, image_range=None, fiji='fiji'):
    """Export a trakem2 project as images.

    Parameters
    ----------
    input : str
        Path to the trakem2 input file.
    output : str
        Path to write subdivided trakem2 inputs.
    image_range :
    fiji : str
        Path to an ImageJ-linux64 executable.
    """
    # On rank 0, create the output directory, get the path to the ImageJ
    # script to run, and split the commands enevly among the ranks.
    if RANK == 0:
        os.makedirs(output, exist_ok=True)
        resource_path = 'ext/export.bsh'
        bsh_path = pkg_resources.resource_filename(__name__, resource_path)
        logging.warning('macro: %s', bsh_path)

        # Select and split a subset of images if a range is passed.
        if not image_range:
            key_sublist = get_keys(input)
        else:
            sub_range = image_range
            keys = np.asarray(range(sub_range[0], sub_range[1]))
            key_sublist = np.array_split(keys, SIZE)

        begin = get_keys(input)[0][0]
    else:
        pass
        key_sublist = None
        bsh_path = None
        begin = None

    # Synchronize and send data to all ranks.
    bsh_path = COMM.bcast(bsh_path, 0)
    key_sublist = COMM.scatter(key_sublist, 0)
    begin = COMM.bcast(begin, 0)

    # Set up the ImageJ command to run.
    print(key_sublist)
    command = '%s -Xms6g -Xmx6g --headless -Dinput=%s -Doutput=%s -Drange=%s -Dbegin=%d -- --no-splash %s' % (
        fiji, input, output, '%d,%d' % (key_sublist[0], key_sublist[-1]), begin, bsh_path)

    # Run the command on each rank.
    print(command)
    os.system(command)


def main():
    args = parse_args()
    mpi_export(args.input,
               args.output,
               args.range,
               fiji=args.fiji)


if __name__ == '__main__':
    main()

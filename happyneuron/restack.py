#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Copy or move a TIFF stack to a new location or CloudVolume layer."""
from __future__ import print_function

import argparse
import glob
import os
import pathlib
import re
import shutil
import sys
import warnings

import numpy as np

from happyneuron.io.img_to_cloudvolume import img2cv


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, default=str(pathlib.Path.home()),
                        help="existing recon foldername")
    parser.add_argument("dest", type=str,
                        help='destination of the copy/move')
    parser.add_argument("--mode", choices=['copy', 'move', 'cloudvolume'],
                        default='copy',
                        help='whether to copy or move the stack, or convert to CloudVolume')
    parser.add_argument("--execute", action='store_true',
                        help='pass to carry out the copy.move action')
    return parser.parse_args(args)


def restack(source, dest, mode='copy', execute=False):
    """Convert a recon stack file tree into a single stack.

    Parameters
    ----------
    source : str
        Top-level directory of the stack to be moved.
    dest : str
        Output directory for the moved/copied stack.
    mode : {'copy', 'move', 'cloudvolume'}
        'copy' will copy the .tiff files to their destination, while 'move'
        move them, emptying the source folders. 'cloudvolume' will create a
        copy and convert it to a CloudVolume layer.
    execute : bool
        If True, copy/move the files. Otherwise, print the source and dest
        paths for each file.
    """

    if mode in ['copy', 'move']:
        os.makedirs(os.path.join(dest, 'full_stack'), exist_ok=True)

        restack_func = shutil.copy if mode == 'copy' else shutil.move

        accum = 0
        for dirname in sorted(glob.glob(os.path.join(source, '*'))):
            globpath = os.path.join(dirname, 'recon*.tif*')
            for fname in sorted(glob.glob(globpath)):
                outfile = os.path.join(dest, 'full_stack',
                                       'recon_{:05d}.tiff'.format(accum))
                print('{} -> {}'.format(fname, outfile))
                if execute:
                    restack_func(fname, outfile)
                accum += 1
    elif mode == 'cloudvolume':
        fnames = []
        for dirname in sorted(glob.glob(os.path.join(source, '*'))):
            globpath = os.path.join(dirname, 'recon*.tif*')
            for fname in sorted(glob.glob(globpath)):
                print('Collected {}'.format(fname))
                fnames.append(fname)
        if execute:
            img2cv(fnames, dest)


def main():
    args = parse_args()
    restack(args.source, args.dest, mode=args.mode, execute=args.execute)


if __name__ == "__main__":
    main()

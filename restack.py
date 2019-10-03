#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function

import argparse
import os
import sys
from os.path import expanduser
import dxchange
import warnings
import numpy as np
from glob import glob
import re, shutil
from tqdm import tqdm


def main(arg):

    parser = argparse.ArgumentParser()
    parser.add_argument("folder_list", help="existing recon foldername")
    parser.add_argument("--new_folder",help="target folder")
    parser.add_argument("--render",help="target folder",default='0')
    args = parser.parse_args()


    prefix = args.prefix
    shift = args.shift
    new_folder = args.new_folder
    
    restack (prefix, new_flder, render)


def restack ( folder_list, new_folder , render):

    print (new_folder)
    try:
        os.makedirs(new_folder)
    except:
        pass

    os.makedirs(os.path.join(new_folder, 'full_stack'))


    accum = 0
    for i, folder in enumerate(tqdm(folder_list)):
        if i < folder_grid.shape[0] - 1:
            shift = shift_ls[i]
        file_list = glob(os.path.join(os.path.join(folder, 'recon', 'recon*.tiff')))
        file_list.sort()
        if i < folder_grid.shape[0] - 1:
            for j, f in enumerate(file_list[:shift]):
                #shutil.copyfile(f, os.path.join(new_folder, 'full_stack', 'recon_{:05d}.tiff'.format(j + accum)))
        accum += 1

if __name__ == "__main__":
    main(sys.argv[1:])
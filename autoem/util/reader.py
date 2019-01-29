from __future__ import print_function

import webbrowser
import numpy as np
import sys,os
import h5py
import dxchange
import argparse
import re

def check_stack_len(f_name):
    matches = re.match(r'^.*/.*(?P<index>[0-9]+).(tif*)$',f_name)
    if matches: 
        #index = int(matches['index'])
        
        #print(match.groups())
        S = int(matches.group(1))
        L = len(os.listdir(os.path.dirname(f_name)))
        #L = len(glob.glob(os.path.dirname(f_name)))
        return (S, L)
    else:
        return (0,1)
def omni_read(f_input, begin=None, end=None):
    '''support tiff, tiff stack, hdf5'''
    if not f_input: return None
    f_input = os.path.abspath(f_input)
    matches = re.match(r'^(?P<dirname>.*)/(?P<fname>.*)\.(?P<ext>[^:]*)($|:(?P<dataset>.*$))', f_input)
    #print(matches.groupdict())
    if matches['ext'] == 'tif' or matches['ext'] == 'tiff':
        if begin is not None and end is not None:
            data = dxchange.read_tiff_stack(f_input, ind=range(begin,end))
        else:
            #print(f_input)
            S,L = check_stack_len(f_input)
            #print(S,L)
            if L > 1:
                data = dxchange.read_tiff_stack(f_input, ind=range(S,S+L))
            else:
                data = dxchange.read_tiff(f_input)
    elif matches['ext'] == 'h5' or matches['ext'] == 'hdf5':
        tokens = f_input.split(':')
        dataset_name = tokens[1]
        f_input = h5py.File(tokens[0], 'r')
        data = np.asarray(f_input[tokens[1]])
    else:
        print('not implemented file type')
    return data

def omni_generator(f_input, begin=0, end=64, step=64):
    for i in range(begin, end, step):
        if i+step > end:
            data = dxchange.read_tiff_stack(f_input, ind=range(i,end))
        else:
            data = dxchange.read_tiff_stack(f_input, ind=range(i,i+step))
        yield (i,data)
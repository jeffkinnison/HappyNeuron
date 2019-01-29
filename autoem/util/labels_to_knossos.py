from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dxchange
import tifffile
#import snappy
import sys
import zipfile
import argparse
import os
import re
from .reader import omni_read
import glob

def labels_to_knossos(f_cube, f_anno, f_ext, mode):
    '''converts segmentated mask back into knossos segmentation channel'''

    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(f_cube)

    #raw = kd.from_raw_cubes_to_matrix(size=kd.boundary, offset=[0, 0, 0])
    print(kd._experiment_name)
    print(kd.boundary)
    if mode == 'to':
        #print(overlay.shape)
        #print(np.unique(overlay[:]))
        new_overlay = omni_read(f_ext)
        new_overlay = np.rollaxis(new_overlay, 2,0)
        new_overlay = np.rollaxis(new_overlay, 2,1)


        print(new_overlay.shape)
        print(np.unique(new_overlay))


        new_kzip = kd.from_matrix_to_cubes(offset=(0,0,0), data=new_overlay, kzip_path=os.path.join(os.path.dirname(f_anno), 'new'))
    elif mode == 'from':
        overlay= kd.from_kzip_to_matrix(path=f_anno,size=kd.boundary, offset=[0, 0, 0], mag=1, verbose=True, alt_exp_name_kzip_path_mode=True)
        print(np.unique(overlay[:]))
        rolled_overlay = np.rollaxis(overlay,2,0)
        rolled_overlay = np.rollaxis(rolled_overlay,2,1)

        f_out = os.path.dirname(f_ext)
        if not os.path.exists(f_out):
            os.makedirs(f_out)
        tifffile.imsave(f_ext, np.uint32(rolled_overlay))







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', default='../full_stack_rot_knossos/mag1/knossos.conf')
    parser.add_argument('--anno', default='../trace/')
    parser.add_argument('--ext', default=None)
    parser.add_argument('--mode','-m', choices=['to','from'], default='to')
    args = parser.parse_args()

    f_cube = args.cube
    curr_anno = glob.glob(os.path.join(args.anno, '*.k.zip'))
    if not curr_anno:
        f_anno = os.path.join(args.anno, 'new.k.zip')
    else:
        f_anno = max(glob.glob(os.path.join(args.anno, '*.k.zip')), key=os.path.getmtime)
    f_ext = args.ext

    #f_swc = os.path.join(f_output, 'output.swc')
    #f_center = os.path.join(f_output, 'center.txt')

    print(f_anno)
    labels_to_knossos(f_cube, f_anno, f_ext, args.mode)


    #sys.exit()
    #print(overlay.shape)
    #print(np.max(overlay[:]))
    #overlay = np.rollaxis(overlay,2,0)
    #overlay = np.rollaxis(overlay,2,1)
    #dxchange.write_tiff(overlay.astype(np.uint32), 'test.tiff', overwrite=True)

if __name__ == '__main__':
    main()
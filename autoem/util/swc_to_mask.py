
from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import numpy as np
import dxchange
import sys
import os
import zipfile
import argparse
import os
import glob
import subprocess
def swc_to_mask(f_cube, f_swc, f_output):
    ''' f_cube: ../mag1/knossos.conf
        f_swc: ../*.swc
        f_output: output dir
    '''
    if not os.path.exists(f_output):
        os.makedirs(f_output)

    f_cube = os.path.abspath(f_cube)
    f_output = os.path.abspath(f_output)
    f_swc = os.path.abspath(f_swc)
    f_tiff_output = os.path.join(f_output, 'stack')

    f_mask = os.path.join(f_output, 'output_mask')
    f_v3d = os.path.join(f_output, 'output_mask.v3draw')
    kd = knossosdataset.KnossosDataset()
    kd.initialize_from_knossos_path(f_cube)
    print(kd._experiment_name)
    print(kd.boundary)
    x,y,z = kd.boundary
    #z,x,y = kd.boundary

    command = r'vaa3d -x swc_to_maskimage_sphere -f swc_to_maskimage -i {} -p {} {} {} -o {}.tiff'.format(f_swc, str(x), str(y), str(z), f_mask)
    print(command)
    #fiji_command = r'run("Convert...", "input={} output={} output_format=TIFF interpolation=Bilinear scale=1"); '.format(f_output, f_output)
    #print(fiji_command)
    #f_fiji = os.path.join(f_output, 'tmp.ijm')
    f_log = os.path.join(f_output, 'log.txt')
    #with open(f_fiji, 'w') as f:
        #f.write(fiji_command+'\n')
    #run_fiji_command = r'fiji -macro {}'.format(f_fiji)
    subprocess.call(command, shell=True)
    #subprocess.call(run_fiji_command, shell=True) 
    #os.system(run_fiji_command)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cube', default=None)
    parser.add_argument('--swc', default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    swc_to_mask(args.cube, args.swc, args.output)
if __name__ == '__main__':
    main()
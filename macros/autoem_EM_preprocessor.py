from __future__ import print_function, division
import os
import glob
import tifffile
import numpy as np
from matplotlib.pyplot import *
import argparse
import shutil
import sys
import cv2
#import magic
import re
from PIL import Image
from ast import literal_eval as make_tuple
from tqdm import tqdm
from pprint import pprint
class EM_preprocessor(object):
    def __init__(self, input_dir, output_dir, stitch, cut_border):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stitch = stitch
        self.ijm_file = None
        self.cut_border = cut_border
        self.flist = None
        self.MAX_ROW = 0
        self.MAX_COL = 0
        self.TILE_ROW = 0
        self.TILE_COL = 0
    def fix_large_pad(self):
        pass
    def prepare_macro(self):
        self.ijm_file = os.path.join(self.output_dir, 'test.ijm')
        commands = []
        with open(self.ijm_file,'w') as f:
            for fname in self.flist:
                #try:
                dirname = fname
                print(dirname)
                os.chdir(dirname)
                tiles = glob.glob('Tile*.tif')
                row = [int(t[6]) for t in tiles]
                col = [int(t[9]) for t in tiles]
                self.MAX_ROW = max(row)
                self.MAX_COL = max(col)
                longer_col = self.MAX_ROW < self.MAX_COL
                min_ind = np.argmin(col if longer_col else row)
                max_ind = np.argmax(col if longer_col else row)
                tilename = tiles[0][0:6]+'{y}'+tiles[0][7:9]+'{x}'+tiles[0][10:]
                command = 'run("Grid/Collection stitching", "type=[Filename defined position] order=[Defined by filename         ] grid_size_x='+str(self.MAX_COL)+' grid_size_y='+str(self.MAX_ROW)+ \
                ' tile_overlap=15 first_file_index_y=1 first_file_index_x=1 directory='+dirname+ \
                ' file_names='+tilename+' output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory='+dirname+'");\n'
                #make sure you have the correct tile_overlap value and correct x and y units for the grid size
                f.write(command)
                min_fname = os.path.join(dirname, tiles[min_ind])
                max_fname = os.path.join(dirname, tiles[max_ind])
                
                print(min_fname,max_fname)
                print(self.cut_border)

                if self.cut_border > 0:
                    if not os.path.exists(min_fname+'.ori'):
                        shutil.copy(min_fname, min_fname+'.ori')
                        shutil.copy(max_fname, max_fname+'.ori')
                        min_data = cv2.imread(min_fname,flags=cv2.IMREAD_GRAYSCALE)
                        max_data = cv2.imread(max_fname,flags=cv2.IMREAD_GRAYSCALE)
                    else:
                        min_data = cv2.imread(min_fname+'.ori',flags=cv2.IMREAD_GRAYSCALE)
                        max_data = cv2.imread(max_fname+'.ori',flags=cv2.IMREAD_GRAYSCALE)
                    print(min_data.shape)
                    if longer_col:
                        print(self.cut_border)
                        #new_min_data = min_data[:,self.cut_border:]
                        #new_max_data = max_data[:,:-self.cut_border]
                        print(min_fname)
                        print(max_fname)
                        cv2.imwrite(min_fname, min_data[:,self.cut_border:])
                        cv2.imwrite(max_fname, max_data[:,:-self.cut_border])
                        pass
                

                    
                    
                #except:
                    #continue
        pass
    def run_fiji(self):
        print('fiji --headless -macro '+self.ijm_file)
        os.system('fiji --headless -macro '+self.ijm_file)
        pass
    def file_too_big(self):
        sample = glob.glob(os.path.join(self.flist[0], 'Tile*'))[0]
        data = cv2.imread(sample,flags=cv2.IMREAD_GRAYSCALE)
        self.TILE_ROW = data.shape[0]
        self.TILE_COL = data.shape[1]
        print(self.TILE_ROW, self.TILE_COL)
        for fname in tqdm(self.flist):
            config_fname = os.path.join(fname, 'TileConfiguration.registered.txt')
            print(config_fname)
            r, c = [],[]
            row_offset,col_offset =[],[]
            tile_names = []
            
            test_output = os.path.join(fname, '')
            try:
                with open(config_fname, 'r') as f:
                    registered = f.readlines()
                    for i in range(4,8): # read offsets
                        tname,_,offsets = registered[i].split(';')
                        _r,_c = int(tname[6])-1, int(tname[9])-1
                        _c_o,_r_o = make_tuple(offsets.strip())
                        r.append(_r)
                        c.append(_c)

                        row_offset.append(np.int32(_r_o))
                        col_offset.append(np.int32(_c_o))

                        tile_names.append(os.path.join(fname, tname))

                    print(row_offset, col_offset)
                    start = tile_names[0].find('S_')
                    output_fname = tile_names[0][start:start+5]+'.tif'
                    print(output_fname)
                    if os.path.exists(output_fname):
                        continue

                    new_data_shape = (max(row_offset)+self.TILE_ROW, max(col_offset)+self.TILE_COL)
                    new_data = np.zeros(new_data_shape, dtype=np.uint8)
                    print(new_data.shape)
                    for i in range(len(tile_names)):
                        _r, _c = r[i], c[i]
                        _r_o, _c_o = row_offset[i], col_offset[i]
                        #print(_r,_c)
                        #print(_r_o,_c_o)
                        #print((_c+1) * self.TILE_COL+_c_o)
                        old_data = cv2.imread(tile_names[i],flags=cv2.IMREAD_GRAYSCALE)
                        new_data[_r_o:_r_o+self.TILE_ROW, \
                                _c_o:_c_o+self.TILE_COL] = old_data
                    cv2.imwrite(os.path.join(self.output_dir,output_fname), new_data)
            except:
                continue
                    



             

                
                
                
    def unify_size(self, x_cut=0, y_cut=0):
        get_index = lambda f: int(f.split('/')[-1].split('_')[1].split('.')[0])
        out_flist = glob.glob(os.path.join(self.output_dir, 'S_*.tif'))
        out_flist.sort(key=get_index)

        cut_out_dir = os.path.join(self.output_dir, 'cut')
        #print(out_flist)
        
        first_flag = True
        for o in out_flist[0:3]:
            old_data = cv2.imread(o,flags=cv2.IMREAD_GRAYSCALE)
            print(old_data.shape)
            if first_flag:
                new_x = (x_cut, old_data.shape[0]-x_cut)
                new_y = (y_cut, old_data.shape[0]-y_cut)
                print(new_x,new_y)
                first_flag = False
            cv2.imwrite(os.path.join(self.output_dir,output_fname), new_data)
            break

    def collect(self):
        collect_dir = os.path.join(self.output_dir, 'prealignment_stack')
        if not os.path.exists(collect_dir):
            os.makedirs(collect_dir)
        if self.stitch:
            flist = glob.glob(os.path.join(self.input_dir, 'S_*/img_t1_z1_c1'))
        else:
            flist = glob.glob(os.path.join(self.input_dir, 'S_*/*.tif'))
        for fname in tqdm(flist):
            #print(fname)
            im = cv2.imread(fname,flags=cv2.IMREAD_GRAYSCALE)
            #print(im.shape)
            RE = re.search('S_([0-9]*)', fname)
            ind = RE.group(1)
            #prefix_len = len(dataDir)

            #outf = outDir+fname[prefix_len:prefix_len+5]+'.tiff'
            outf = os.path.join(collect_dir, 'S_'+ind+'.tiff')
            #print(outf)
            tifffile.imsave(outf,im)


    def prepare_align_txt(self):
        collect_dir = os.path.join(self.output_dir, 'prealignment_stack')
        f_align_txt = os.path.join(self.output_dir, 'align.txt')
        
        #flist = np.asarray(glob.glob(os.path.join(collect_dir, '*.tiff')))
        #f_order = np.argsort([int(f.split('/')[-1][2:5]) for f in flist]).tolist()
        #flist = flist[f_order]
        
        flist = np.asarray(glob.glob(os.path.join(collect_dir, '*.tif*')))
        inds = [int(re.search('.*_([0-9]*)', f.split('/')[-1]).group(1)) for f in flist]
        flist = flist[np.argsort(inds)]
        with open(f_align_txt, 'w') as output:
        #output = open(outFile, 'w')
            for i in range(len(flist)):
                #print(flist[i],0,0,i)
                command = flist[i]+'\t'+'0\t'+'0\t'+str(i)+'\n'
                print(command)
                output.write(command)
            output.close() 

        #print(flist)

        pass
    def run(self):
        print("Input:", self.input_dir)
        print("Output:", self.output_dir)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.flist= glob.glob(os.path.join(self.input_dir,'S_*'))
        get_index = lambda f: int(f.split('/')[-1].split('_')[1])
        self.flist.sort(key=get_index)
        #pprint(self.flist)
        
        if self.stitch:
            # Step 1: Prepare stitch macro, fix large image
            self.prepare_macro()
            
            # Step 2: Run stitch
            self.run_fiji()
        
            # Step 3: (Optional) if stitched data larger than 2^31 pixels
            #self.file_too_big()
            # Unify size
            #self.unify_size()
        
        # Step 4: collect fiji stitched results into output folder
        self.collect()
        
        # Step 5: prepare TrackEM2 import txt file
        #self.prepare_align_txt(self.input_dir, self.input_dir)
        self.prepare_align_txt()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--stitch', default=False, type=bool)
    parser.add_argument('--cut_border')
    args = parser.parse_args()
    rootDir = os.getcwd()
    #dataDir ='/home/vandana/Desktop/Section_Set_4/'
    cut_border = 0 if not args.cut_border else int(args.cut_border)
    emp = EM_preprocessor(args.input, args.output, args.stitch, cut_border)
    emp.run()
    
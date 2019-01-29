'''Preview for ATLAS EM images.'''
import os, errno
import argparse
import cv2
import glob
from tqdm import tqdm

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument('--input_dir', default=None, type=str)
 parser.add_argument('--output_dir', default='ds', type=str)
 parser.add_argument('--size', default=1024, type=int)
 parser.add_argument('--offset', default=None, type=str)
 args = parser.parse_args()

 tiles = glob.glob(os.path.join(args.input_dir, 'S_*/*.tif*'))
 tiles.sort()


 if args.offset == None:
   im0 = cv2.imread(tiles[0], 0)
   offset_x = im0.shape[0]   / 2
   offset_y = im0.shape[1]   / 2

 print('Number of Tiles: ', len(tiles))
 print('Preview Size: ', args.size)
 print('Preview offset_x', offset_x-args.size/2)
 print('Preview offset_y', offset_y-args.size/2)
 os.mkdir(args.output_dir)
 print ('Saving at '+os.path.join(os.getcwd(),args.output_dir))

 for t in tqdm(tiles):
   filename = os.path.basename(t)
   f_out = os.path.join(args.output_dir, filename)
   f_out = os.path.splitext(f_out)[0]+'.jpeg'
   im = cv2.imread(t, 0)
   im_down = im[offset_x-args.size/2:offset_x+args.size/2,offset_y-args.size /2:offset_y+args.size/2]
   cv2.imwrite(f_out, im_down)

if __name__ == '__main__':
 main()
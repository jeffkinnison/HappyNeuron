import os
import argparse
import glob 
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import netpbmfile
import tifffile


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mask_dir', default='masks', type=str)
  parser.add_argument('--output_mask', default=None, type=str)
  args = parser.parse_args()
  f_list = glob.glob(os.path.join(args.mask_dir, '*.pbm*'))
  f_0 = cv2.imread(f_list[0], 0)
  merged_mask = np.zeros(f_0.shape, dtype=np.int32)

  for f in tqdm(f_list):
    mask = cv2.imread(f, 0)
    merged_mask += mask.astype(np.int32) // 255

  #cv2.imwrite(args.output_mask, merged_mask)
  tifffile.imsave(args.output_mask, merged_mask, dtype=np.int32)
  print('done')

if __name__ == '__main__':
  main()

''' Cut the image dim to multiples of 8. '''
import os
import argparse
import glob 
import numpy as np
import cv2
from tqdm import tqdm

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', default=None, type=str)
  parser.add_argument('--output_dir', default=None, type=str)
  args = parser.parse_args()
  os.makedirs(args.output_dir, exist_ok=True)
  # is os.path.exists(args.output_dir):

    
  for f_input in tqdm(glob.glob(os.path.join(args.input_dir, '*.tif*'))):
    bname = os.path.basename(f_input)
    f_output = os.path.join(args.output_dir, bname)

    img = cv2.imread(f_input, 0)
    # print(img.shape)
    q, r = np.divmod(img.shape, 8)
    # if r == [0, 0]:
    #   break
    r_range = (r[0] // 2, q[0] * 8 + r[0] // 2)
    c_range = (r[1] // 2, q[1] * 8 + r[0] // 2)
    
    cv2.imwrite(f_output, img[r_range[0]:r_range[1], c_range[0]:c_range[1]])
    # print(q, r)
    # break

    # mask_img = Image.fromarray(mask)
    # with open(f_mask, 'wb') as f_m:
    #   mask_img.save(f_m)


if __name__ == '__main__':
  main()
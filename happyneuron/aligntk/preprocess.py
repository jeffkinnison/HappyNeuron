import os
import argparse
import glob 
import re
import numpy as np
from pprint import pprint

def preprocess_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None, type=str)
  parser.add_argument('--n_groups', default=12, type=int)
  args = parser.parse_args()
  image_dir = args.image_dir
  output_dir = args.output_dir
  n_groups = args.n_groups
  return image_dir, output_dir, n_groups

def get_index(filename):
  """Get index of a **/*.tif filename"""
  return int(re.search(r'(^\d)*(\d+).*', filename).group(2))

def preprocess_main(image_dir, output_dir, n_groups):
  # Write complete set
  os.makedirs(output_dir, exist_ok=True)
  f_images = os.path.join(output_dir, 'images.lst')
  f_pairs = os.path.join(output_dir, 'pairs.lst')
  image_list = [s.split('.')[0] for s in glob.glob1(image_dir, '*.tif*')]
  image_list.sort()

  image_dict = {}
  pairs_dict = {}
  with open(f_images, 'w') as f1:
    for im in image_list:
      image_dict[get_index(im)] = im
      f1.write(im+'\n')


  with open(f_pairs, 'w') as f2:
    for i in range(len(image_list)-1):
      curr_idx = get_index(image_list[i])
      next_idx = get_index(image_list[i+1])
      if next_idx - curr_idx == 1:
        line = image_list[i] + ' ' + image_list[i+1] + ' ' + \
          image_list[i] + '_' + image_list[i+1]
        pairs_dict[(curr_idx, next_idx)] = line
        f2.write(line + '\n')

  # Write individual groups
  n_pairs = len(image_list) - 1
  pair_keys = sorted(pairs_dict.keys())

  split_pair_keys = np.array_split(pair_keys, n_groups)

  for group_id, group_pair_keys in enumerate(split_pair_keys):
    f_images = os.path.join(output_dir, 'images%d.lst' % group_id)
    f_pairs = os.path.join(output_dir, 'pairs%d.lst' % group_id)

    with open(f_images, 'w') as f1:
      [f1.write(image_dict[k]+'\n') for k in group_pair_keys.ravel()]

    with open(f_pairs, 'w') as f2:
      [f2.write(pairs_dict[tuple(k)]+'\n') for k in group_pair_keys]
  

def main():
  image_dir, output_dir, group_size = preprocess_parse()
  
  preprocess_main(image_dir, output_dir, group_size)
    


if __name__ == '__main__':
  main()

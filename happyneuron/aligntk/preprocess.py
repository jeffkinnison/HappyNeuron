import os
import argparse
import glob 
from pprint import pprint


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None, type=str)
  # parser.add_argument('--groups', default=1, type=int)
  parser.add_argument('--group_size', default=12, type=int)
  args = parser.parse_args()

  # Write complete set
  f_images = os.path.join(args.output_dir, 'images.lst')
  f_pairs = os.path.join(args.output_dir, 'pairs.lst')
  image_list = [s.split('.')[0] for s in glob.glob1(args.image_dir, '*.tif*')]
  image_list.sort()
  with open(f_images, 'w') as f1:
    [f1.write(im+'\n') for im in image_list]

  with open(f_pairs, 'w') as f2:
    for i in range(len(image_list)-1):
      f2.write(image_list[i] + ' ' + image_list[i+1] + ' ' 
        + image_list[i] + '_' + image_list[i+1] + '\n')

  # Write individual groups
  image_sets = {}
  pairs_sets = {}
  groups = len(image_list)  // args.group_size + 1

  
  # set_size = len(image_list) // args.groups + 1

  for i in range(groups):
    f_images = os.path.join(args.output_dir, 'images%d.lst' % i)
    f_pairs = os.path.join(args.output_dir, 'pairs%d.lst' % i)
    if i == 0:
      image_sets[i] = [im for im in image_list[i*args.group_size:(i+1)*args.group_size]]
    else:
      image_sets[i] = [im for im in image_list[i*args.group_size-1:(i+1)*args.group_size]]

    with open(f_images, 'w') as f1:
      [f1.write(im+'\n') for im in image_sets[i]]

    with open(f_pairs, 'w') as f2:
      for j in range(len(image_sets[i])-1):
        f2.write(image_sets[i][j] + ' ' + image_sets[i][j+1] + ' ' 
          + image_sets[i][j] + '_' + image_sets[i][j+1] + '\n')

  pprint(image_sets)
    


if __name__ == '__main__':
  main()

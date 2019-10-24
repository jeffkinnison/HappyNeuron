import os
import argparse
import glob 
from pprint import pprint


def preprocess_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output_dir', default=None, type=str)
  parser.add_argument('--group_size', default=12, type=int)
  image_dir = args.image_dir
  output_dir = args.output_dir
  group_size = args.group_size
  return image_dir, output_dir, group_size
  
def preprocess_main(image_dir, output_dir, group_size):
  # Write complete set
  f_images = os.path.join(output_dir, 'images.lst')
  f_pairs = os.path.join(output_dir, 'pairs.lst')
  image_list = [s.split('.')[0] for s in glob.glob1(image_dir, '*.tif*')]
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
  groups = len(image_list)  // group_size + 1

  # set_size = len(image_list) // args.groups + 1

  for i in range(groups):
    f_images = os.path.join(output_dir, 'images%d.lst' % i)
    f_pairs = os.path.join(output_dir, 'pairs%d.lst' % i)
    if i == 0:
      image_sets[i] = [im for im in image_list[i*group_size:(i+1)*group_size]]
    else:
      image_sets[i] = [im for im in image_list[i*group_size-1:(i+1)*group_size]]

    with open(f_images, 'w') as f1:
      [f1.write(im+'\n') for im in image_sets[i]]

    with open(f_pairs, 'w') as f2:
      for j in range(len(image_sets[i])-1):
        f2.write(image_sets[i][j] + ' ' + image_sets[i][j+1] + ' ' 
          + image_sets[i][j] + '_' + image_sets[i][j+1] + '\n')

  pprint(image_sets)

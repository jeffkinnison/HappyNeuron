
import os
import argparse
import glob 
import numpy as np
import cv2
from scipy import ndimage
from tqdm import tqdm
from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def gen_triplets(f_list):
  print(f_list)
  return [ [f_list[i-1], f_list[i], f_list[i+1]] for i in range(1, len(f_list)-1) ]
  # return [ [f_list[i-1], f_list[i], f_list[i+1]] for i, _ in enumerate(f_list[1:-1]) ]

def process_triplets(f_triplet, threshold=0):
  vol = np.asarray([cv2.imread(f, 0) for f in f_triplet])

  # background
  img = vol[1,...]
  mask = np.less_equal(img, threshold)
  labels, _ = ndimage.label(mask)
  ids, counts = np.unique(labels, return_counts=True)
  max_id = ids[1:][np.argmax(counts[1:])]
  bg_mask = labels == max_id
  print('max', max_id)
  
  # background + defects
  new_vol = np.copy(vol)
  full_mask = np.less_equal(img, threshold).astype(np.uint8)

  # defects
  defect_mask = np.logical_and(full_mask, np.logical_not(bg_mask))
  
  kernel = np.ones((3,3), np.uint8)
  defect_mask = cv2.dilate(defect_mask.astype(np.uint8), kernel, 1) == 1

  if np.mean(new_vol[0]) > np.mean(new_vol[2]):
    new_vol[1,defect_mask] = new_vol[0,defect_mask]
  else:
    new_vol[1,defect_mask] = new_vol[2,defect_mask]
  
  return new_vol[1,...]


def mpi_triplet_process(input_dir, output_dir, process_fn, params={}):
  '''MPI process

  Args:
    input_dir: input directory containing images
    output_dir: output directory
    process_fn: A function taking (img, **params) as input arguments
    params: additional parameters for process_fn
  '''
  if mpi_rank == 0:
    f_list = glob.glob(os.path.join(input_dir, '*.*'))
    f_list.sort()
    f_triplet_list = gen_triplets(f_list)
    print(f_triplet_list)
    f_sublist = np.array_split(np.asarray(f_triplet_list), mpi_size)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
  else:
    f_sublist = None

  f_sublist = mpi_comm.scatter(f_sublist, 0)

  for f in tqdm(f_sublist):
    img_out = process_fn(f, params['threshold'])
    f_out = os.path.join(output_dir, os.path.basename(f[1]))
    cv2.imwrite(f_out, img_out)
    # img = cv2.imread(f, 0)
    # img_out = process_fn(img, **params)
    # f_out = os.path.join(output_dir, os.path.basename(f))
    # cv2.imwrite(f_out, img_out)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--threshold', default=0, type=int)
  args = parser.parse_args()

  params = dict(threshold=args.threshold)
  mpi_triplet_process(args.input, args.output, process_triplets, params)



if __name__ == '__main__':
  main()

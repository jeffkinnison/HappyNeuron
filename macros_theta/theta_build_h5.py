import h5py
from skimage import io
import argparse
import os
from os.path import exists, join, dirname, basename
import glob
import re
import numpy as np
from tqdm import tqdm
from pprint import pprint

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def image_gen(f_list):
  for i, f in enumerate(f_list):
    yield i, io.imread(f)

def write_h5(image_dir, label_dir=None, output=None, axes='zyx'):
  assert exists(image_dir)
  os.makedirs(dirname(output), exist_ok=True)
  f_list = glob.glob(join(image_dir, '*.*'))
  f_list.sort()
  z = len(f_list)
  test_im = io.imread(f_list[0])
  x, y = test_im.shape
  print(test_im.shape)
  
  with h5py.File(output, 'w') as f:
    img_ds = f.create_dataset('image', shape=(z,y,x), dtype=np.uint8, chunks=(32, 128, 128))
    for i, im in tqdm(image_gen(f_list)):
      img_ds[i, :, :] = np.transpose(im)
      if i >= 32:
        break
def get_z(f_path):
  return int(re.search(r'(\d+)\..*', f_path).group(1))

def mpi_write_h5(image_dir, output, label_dir=None, axes='zyx'):
  assert exists(image_dir)
  output_dir = dirname(output)
  os.makedirs(output_dir, exist_ok=True)

  if mpi_rank == 0:
    f_list = glob.glob(os.path.join(image_dir, '*.*'))
    f_list.sort()
    f_sublist = np.array_split(np.asarray(f_list), mpi_size)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
    z = len(f_list)
    test_im = io.imread(f_list[0])
    x, y = test_im.shape
    print(z, y, x)
  else:
    f_sublist = None
    z = None
    y = None
    x = None

  f_sublist = mpi_comm.scatter(f_sublist, 0)
  z = mpi_comm.bcast(z, 0)
  y = mpi_comm.bcast(y, 0)
  x = mpi_comm.bcast(x, 0)

  with h5py.File(output, 'w', driver='mpio', comm=mpi_comm) as f:
    img_ds = f.create_dataset('image', shape=(z,y,x), dtype=np.uint8)
    for f in tqdm(f_sublist):
      curr_z = get_z(f)
      img = np.transpose(io.imread(f))
      img_ds[curr_z, :, :] = img
  pass
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_dir', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--label_dir', default=None, type=str)
  args = parser.parse_args()
  mpi_write_h5(args.image_dir, output=args.output, label_dir=None, )
if __name__ == '__main__':
  main()

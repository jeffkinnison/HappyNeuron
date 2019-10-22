'''Preview for a stack of EM images.'''
import os
import errno
import argparse
import numpy as np
import cv2
import glob
from tqdm import tqdm

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def mpi_run(tile_list, output_dir, offset, preview_size):
  for t in tqdm(tile_list):
    filename = os.path.basename(t)
    f_out = os.path.join(output_dir, filename)
    f_out = os.path.splitext(f_out)[0]+'.jpeg'
    im = cv2.imread(t, 0)
    im_down = im[offset[0]-preview_size // 2:offset[0]+preview_size // 2,
                 offset[1]-preview_size // 2:offset[1]+preview_size // 2]
    cv2.imwrite(f_out, im_down)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default='./preview', type=str)
  parser.add_argument('--size', default=1024, type=int)
  parser.add_argument('--offset', default=None, type=str)
  args = parser.parse_args()

  if mpi_rank == 0:
    os.makedirs(args.output, exist_ok=True)
    tiles = glob.glob(os.path.join(args.input, '*.tif*'))
    tiles.sort()
    tile_subset = np.array_split(np.asarray(tiles), mpi_size)

    if args.offset == None:
      im0 = cv2.imread(tiles[0], 0)
      offset = [i // 2 for i in im0.shape]
      # offset_x = im0.shape[0]   // 2
      # offset_y = im0.shape[1]   // 2

    print('Number of Tiles: ', len(tiles))
    print('Preview Size: ', args.size)
    print('Preview offset_x', offset[0]-args.size//2)
    print('Preview offset_y', offset[1]-args.size//2)
    # os.mkdir(args.output_dir)
    print('Saving at '+os.path.join(os.getcwd(), args.output))
  else:
    tile_subset = None
    offset = None
  tile_subset = mpi_comm.scatter(tile_subset, root=0)
  offset = mpi_comm.bcast(offset, root=0)

  print(tile_subset)
  mpi_run(tile_subset, args.output, offset, args.size)


if __name__ == '__main__':
  main()

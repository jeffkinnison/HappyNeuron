import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process

def cut_image(img, start_xy, size_xy, angle=0.0):
  X, Y = img.shape
  if angle != 0.0:
    rot_mat = cv2.getRotationMatrix2D((Y/2, X/2), angle, 1)
    img = cv2.warpAffine(img, rot_mat, (Y, X))
  sx, sy = start_xy
  dx, dy = size_xy
  padd_x, padd_y = 0, 0
  if dx > X or dy > Y:
    padd_x = max(0, dx-X)
    padd_y = max(0, dy-Y)
    img = np.pad(img, ((0,padd_x), (0,padd_y)), mode='constant')
  return img[sx:sx+dx, sy:sy+dy]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--start_xy', default='0,0', type=str)
  parser.add_argument('--size_xy', default='1024,1024', type=str)
  parser.add_argument('--angle', default=0.0, type=float, 
    help='counter clockwise degrees')
  args = parser.parse_args()

  sx, sy  = [int(i) for i in args.start_xy.split(',')]
  dx, dy =  [int(i) for i in args.size_xy.split(',')]
  dx = dx // 8 * 8
  dy = dy // 8 * 8

  params = dict(start_xy=(sx, sy), size_xy=(dx, dy), angle=args.angle)
  mpi_process(args.input, args.output, cut_image, params)


if __name__ == '__main__':
  main()

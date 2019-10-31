import os
import argparse
import numpy as np
import cv2
import imutils
import glob
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process


def get_min_rect(img, kernel, iterations):
  ret,thresh = cv2.threshold(img, 0.5, 255, 0)
  kernel = np.ones((kernel, kernel), np.uint8)
  thresh = cv2.dilate(thresh, kernel, iterations=iterations)
  contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  x, y, w, h = cv2.boundingRect(contours[0])
  return img[y:y+h, x:x+w]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--kernel', default=3, type=int)
  parser.add_argument('--iterations', default=4, type=int)
  args = parser.parse_args()

  params = dict(kernel=args.kernel, iterations=args.iterations)
  mpi_process(args.input, args.output, get_min_rect, params)


if __name__ == '__main__':
  main()

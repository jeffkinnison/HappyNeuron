import os
import argparse
import numpy as np
import cv2
import imutils
import glob
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process

def clahe_image(image, clipLimit=2.0, tileGridSize=(8,8)):
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  return clahe.apply(image)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--clip', default=2.0, type=float)
  parser.add_argument('--grid', default=8, type=int)
  args = parser.parse_args()

  params = dict(clipLimit=args.clip, tileGridSize=(args.grid, args.grid))
  mpi_process(args.input, args.output, clahe_image, params)


if __name__ == '__main__':
  main()

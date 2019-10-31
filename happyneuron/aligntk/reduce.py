import os
import argparse
import numpy as np
import cv2
import glob
from PIL import Image
from tqdm import tqdm
from .utils import mpi_process
from .utils import reduce_size


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('input', default=None, type=str)
  parser.add_argument('output', default=None, type=str)
  parser.add_argument('--factor', default=2, type=int)
  args = parser.parse_args()

  mpi_process(args.input, args.output, reduce_size,  dict(factor=args.factor))


if __name__ == '__main__':
  main()

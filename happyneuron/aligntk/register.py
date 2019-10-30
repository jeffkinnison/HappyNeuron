import struct
from pprint import pprint
import glob
import os
import subprocess
import numpy as np
import argparse
import re
from tqdm import tqdm

def preprocess_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--aligntk_path', default=None, type=str)
  parser.add_argument('--pairs', default=None, type=str)
  parser.add_argument('--images', default=None, type=str)
  parser.add_argument('--mask', default=None, type=str)
  parser.add_argument('--output', default=None, type=str)
  parser.add_argument('--distortion', default=None, type=str)
  parser.add_argument('--output_level', default=None, type=str)
  parser.add_argument('--depth', default=None, type=str)
  parser.add_argument('--quality', default=None, type=str)
  parser.add_argument('--summary', default=None, type=str)
  parser.add_argument('--initial_map', default=None, type=str)
  args = parser.parse_args()
  return args

def register(
    aligntk_path,
    pairs,
    images,
    mask,
    output,
    distortion,
    output_level,
    depth,
    quality,
    summary,
    initial_map):
    """ Wrapper for find_rst in aligntk. 
    Args:
        aligntk_path: Path to aligntk bins
        pairs: Path to a *.lst file with lines in form of "{src} {tgt} {src}_{tgt}"
        images: Path to a *.lst file with lines in form of image names
        mask: Dir with mask images
        output: Output dir
        distortion:
        output_level:
        depth:
        quality:
        summary: summary dir
        initial_map:
    """
    command = '{}/register -pairs {} -tif -images {} -mask {} -output {} -distortion {} -output_level {} -depth {} -quality {} -summary {} -initial_map {}'.format(
            aligntk_path,
            pairs,
            images,
            mask,
            output,
            distortion,
            output_level,
            depth,
            quality,
            summary,
            initial_map)
            
    os.system(command)


def main():
  args = preprocess_parse()
  register(**vars(args))


if __name__ == '__main__':
    main()

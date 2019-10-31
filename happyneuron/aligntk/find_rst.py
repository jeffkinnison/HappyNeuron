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
  parser.add_argument('--rotation', default=None, type=str)
  parser.add_argument('--max_res', default=None, type=str)
  parser.add_argument('--scale', default=None, type=str)
  parser.add_argument('--tx', default=None, type=str)
  parser.add_argument('--ty', default=None, type=str)
  parser.add_argument('--summary', default=None, type=str)
  args = parser.parse_args()
  return args

def find_rst(
    aligntk_path,
    pairs,
    images,
    mask,
    output,
    rotation,
    max_res,
    scale,
    tx,
    ty,
    summary): 
    """ Wrapper for find_rst in aligntk. 
    Args:
        aligntk_path: Path to aligntk bins
        pairs: Path to a *.lst file with lines in form of "{src} {tgt} {src}_{tgt}"
        images: Path to a *.lst file with lines in form of image names
        mask: Dir with mask images
        output: Output dir
        rotation: Tuple of ints, specifying allowed rotation range
        max_res: Max resolution
        scale: Tuple of ints, specifying allowed scale range
        tx: X-translation allowed range
        ty: Y-translation allowed range
        summary: summary dir
    """
    command = '{}/find_rst -pairs {} -tif -images {} -mask {} -output {} -rotation {} -max_res {} -scale {} -tx {} -ty {} -summary {}'.format(
            aligntk_path,
            pairs,
            images,
            mask,
            output,
            rotation,
            max_res,
            scale,
            tx,
            ty,
            summary)
            
    os.system(command)


def main():
  args = preprocess_parse()
  find_rst(**vars(args))


if __name__ == '__main__':
    main()

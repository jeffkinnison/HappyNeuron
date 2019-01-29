from knossos_utils import knossosdataset, skeleton
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import dxchange
import tifffile
#import snappy
import sys
import zipfile
import argparse
import os
import re
from .reader import omni_read
import glob

def labels_to_skeleton(labels):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=str)
    args = parser.parse_args()

    labels = omni_read(args.labels)
    labels_to_skeleton(labels)

if __name__ == '__main__':
    main()
"""Preview cutouts of ATLAS EM images."""

import argparse
import errno
import glob
import os

import cv2
from tqdm import tqdm


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None, type=str,
                        help='path to the images to preview')
    parser.add_argument('--output_dir', default='ds', type=str,
                        help='path to write preview images to')
    parser.add_argument('--size', default=1024, type=int,
                        help='shape of the preview')
    parser.add_argument('--offset', default=None, type=int, nargs=2,
                        help='offset into the image from pixel (0, 0)')
    args = parser.parse_args()


def preview(input_dir, output_dir='ds', size=1024, offset=None):
    """Preview cutouts of ATLAS EM images.

    Parameters
    ----------
    input_dir : str
        Path to the images to preview.
    output_dir : str
        Path to write preview images to.
    size : int
        Shape of the preview cutout.
    offset : tuple, optional
        Offset into the image from pixel (0, 0).
    """
    # Get the images to preview.
    tiles = glob.glob(os.path.join(input_dir, 'S_*/*.tif*'))
    tiles.sort()

    # If no offset was provided, assume that the cutout should be a square
    # around the center of the image.
    if args.offset == None:
        im0 = cv2.imread(tiles[0], 0)
        offset_x = im0.shape[0] / 2
        offset_y = im0.shape[1] / 2

    print('Number of Tiles: ', len(tiles))
    print('Preview Size: ', size)
    print('Preview offset_x', offset_x-size/2)
    print('Preview offset_y', offset_y-size/2)
    os.makedirs(output_dir, exist_ok=True)
    print('Saving at {}'.format(os.path.join(os.getcwd(), output_dir)))

    # Index into each image in the stack, cut out the preview, and write the
    # cutout to file.
    size = size / 2
    startx, endx = offset_x - size, offset_x + size
    starty, endy = offset_y - size, offset_y + size
    for t in tqdm(tiles):
        filename = os.path.basename(t)
        f_out = os.path.join(output_dir, filename)
        f_out = os.path.splitext(f_out)[0]+'.jpeg'
        im = cv2.imread(t, 0)
        im_down = im[startx:endx, starty:endy]
        cv2.imwrite(f_out, im_down)


def main():
    args = parse_args()
    preview(args.input_dir,
            output_dir=args.output_dir,
            size=args.size,
            offset=args.offset)


if __name__ == '__main__':
    main()

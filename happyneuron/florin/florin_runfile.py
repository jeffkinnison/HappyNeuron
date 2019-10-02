import argparse

import florin
import florin.conncomp
import florin.morphology
import numpy as np


def parse_args(args=None):
    p = argparse.ArgumentParser(
        description='Run a neural volume through FLoRIN.')

    p.add_argument('--input', type=str, help='path to the input volume')
    p.add_argument('--output', type=str, help='path to write the segmentation')
    p.add_argument('--tile-shape', type=int, nargs='*',
                   help='shape of subvolumes to process in parallel')
    p.add_argument('--tile-stride', type=int, nargs='*',
                   help='offset between subvolumes to process in parallel')
    p.add_argument('--ndnt-shape', type=int, nargs='*',
                   help='shape of the pixel neighborhood window in NDNT')
    p.add_argument('--ndnt-threshold', type=float,
                   help='threshold value in [0, 1], lower thresholds stronger')

    return p.parse_args(args)


def florin_xray(input, output, tile_shape, tile_stride, ndnt_shape, threshold):
    """Segment a neural volume with FLoRIN.

    Parameters
    ----------
    input : str
        Path to the input image data.
    output : str
        Path to write the segmentation to.
    tile_shape : tuple of int
        Shape of subvolumes to process in parallel, specifies the shape along
        the (z, y, x) axes, e.g. (10, 64, 64).
    tile_stride : tuple of int
        Spacing between subvolumes processed in parallel, allowing for overlap.
        Specifies the step along the (z, y, x) axes between the first entry of
        each subvolume, e.g. (8, 48, 48)
    ndnt_shape : tuple of int
        Shape of the NDNT sliding window used to compute pixel neighborhood
        statistics. Specifies the shape along the (z, y, x) axes, e.g.
        (4, 8, 8).
    threshold : float
        The threshold value determining the strength of NDNT, must be in
        ``[0, 1]``. Lower values indicate stronger thresholds.
    """
    load_op = florin.load()
    sel = np.zeros((3, 3, 3), dtype=np.uint8)
    sel[1] = 1
    pipe = florin.SerialPipeline(
        load_op,
        florin.tile(shape=tile_shape, stride=tile_stride),
        florin.MPITaskQueue(
            florin.ndnt(shape=ndnt_shape, threshold=threshold),
            florin.morphology.binary_erosion(structure=sel),
            florin.morphology.binary_opening(structure=sel),
            florin.morphology.remove_small_holes()
        )
        florin.conncomp.label(connectivity=2),
        florin.morphology.remove_small_objects(min_size=100),
        florin.save(output)
    )
    pipe.run(input)


def main():
    args = parse_args()
    florin_xray(args.input, args.output, args.tile_shape, args.tile_stride,
                args.ndnt_shape, args.threshold)


if __name__ == '__main__':
    main()

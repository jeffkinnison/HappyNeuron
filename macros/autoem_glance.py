


import numpy as np

from autoem.util import neuroglance




if __name__ == '__main__':

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--profile', '-p',
        help="An experiment profile in toml. If no file is specified, ",
        default=None
    )

    f_input = '/mnt/md0/XRay/2017_10_16_VS547/masked_stack/image.tif'
    f_labels = '/mnt/md0/XRay/2017_10_16_VS547/masked_stack/prediction/19/cc/dendrite_cc.tif'
    raw = np.asarray(dxchange.read_tiff(f_input))
    labels = np.uint32(dxchange.read_tiff(f_labels) > 128)

    viewer = neuroglancer.Viewer()
    url = glance(viewer=viewer, raw=raw, labels=labels)
    webbrowser.open_new(url)

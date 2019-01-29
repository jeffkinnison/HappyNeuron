from __future__ import print_function

import webbrowser
import numpy as np
import sys
import h5py
import neuroglancer
import dxchange
import argparse

#from PyQt5.QtWidgets import *
#print(labels)


## subvol
# f_input = h5py.File('../masked_stack/train-input.h5', 'r')
# f_labels = h5py.File('../masked_stack/train-labels.h5', 'r')
# raw = f_input['raw']
# labels = f_labels['stack']
## full vol
#raw = dxchange.read_tiff_stack('../full_stack_rot/recon_0000.tif', ind=range(0,1920))

def glance(viewer, raw, labels=None):
    with viewer.txn() as s:
        s.voxel_size = [600, 600, 600]
        s.layers.append(
            name='image',
            layer=neuroglancer.LocalVolume(
                data=raw,
                offset = (0,0,0),
                voxel_size = s.voxel_size,
            ),
            shader="""
    void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                toNormalized(getDataValue(1)),
                toNormalized(getDataValue(2))));
    }
    """),
        if labels is not None:
            s.layers.append(
                name='labels',
                layer=neuroglancer.LocalVolume(
                    data=labels,
                    offset = (0,0,0),
                    voxel_size = s.voxel_size,
                ),
            )
    return viewer.get_viewer_url()
#class neuroglancerUI()
#def main():
#    APP = QApplication()
#    WINDOW = QDialog()
#    UI = KnossosCuberUI(WINDOW, APP, CONFIG)
#    UI.update_gui_from_config()
#
#    WINDOW.show()
#
#    sys.exit(APP.exec_())
if __name__ == '__main__':
    #f_input = h5py.File('../ffn/third_party/neuroproof_examples/validation_sample/grayscale_maps.h5', 'r')
    #f_labels = h5py.File('../ffn/third_party/neuroproof_examples/validation_sample/groundtruth.h5','r')
    #raw = f_input['raw']
    #labels = np.uint32(f_labels['stack'])
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


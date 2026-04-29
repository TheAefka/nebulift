"""High level processes for MTObjects."""
# TODO rename?

import sys
import numpy as np
from mtolib import _ctype_classes as ct
from mtolib.preprocessing import preprocess_image
from mtolib import maxtree
from mtolib.tree_filtering import filter_tree, get_c_significant_nodes, init_double_filtering
from mtolib.io_mto import generate_image, generate_parameters, read_fits_file, read_fits_file2, make_parser
from mtolib.utils import time_function
from ctypes import c_float, c_double
from mtolib.postprocessing import relabel_segments
from matplotlib import pyplot as plt


def setup():
    """Read in a file and parameters; run initialisation functions."""

    # Parse command line arguments
    p = make_parser().parse_args()

    # Warn if using default soft bias
    if p.soft_bias is None:
        p.soft_bias = 0.0

    if p.cosmos >= 0:
        img, psf = read_fits_file2(p.filename, p.cosmos)
        print(img.shape)
        print(psf.shape)

        """
        plt.figure(figsize=(8, 6))
        plt.imshow(psf, cmap='gray', origin='lower')
        plt.title('PSF Image')
        plt.colorbar(label='Intensity')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.imshow(np.load('mock/psf.npy'), cmap='gray', origin='lower')
        plt.title('PSF Image')
        plt.colorbar(label='Intensity')
        plt.show()
        #"""
    else:
        img = read_fits_file(p.filename)
        psf = None

    if p.verbosity:
        print("\n---Image dimensions---")
        print("Height = ", img.shape[0])
        print("Width = ", img.shape[1])
        print("Size = ", img.size)

    # Set the pixel type based on the type in the image
    p.d_type = c_float
    if np.issubdtype(img.dtype, np.float64):
        p.d_type = c_double
        init_double_filtering(p)

    # Initialise CTypes classes
    ct.init_classes(p.d_type)

    return img, psf, p


def max_tree_timed(img, params, maxtree_class):
    """Build and return a maxtree of a given class"""
    if params.verbosity:
        print("\n---Building Maxtree---")
    mt = maxtree_class(img, params.verbosity, params)
    mt.flood()
    return mt


def build_max_tree(img, params, maxtree_class=maxtree.OriginalMaxTree):
    return time_function(max_tree_timed, (img, params, maxtree_class), params.verbosity, 'create max tree')

def biggest_map(image, id_map, biggest):
    big_map = np.zeros(image.shape) -1
    for i in range(len(id_map)):
        for j in range(len(id_map[i])):
            if id_map[i][j] == biggest:
                big_map[i][j] = biggest
    return big_map
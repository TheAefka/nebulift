from argparse import Namespace
from ctypes import c_float, c_double
import numpy as np

import mtolib.main as mto
from mtolib.io_mto import read_fits_file
from mtolib import _ctype_classes as ct_classes

def run_mto(
        fits_path: str,
        out_path: str = 'output.png',
        param_out_path: str = 'parameters.csv',
        alpha: float = 1e-6,
        bg_mean: float | None = None,
        bg_variance: float = -1.0,
        gain: float = -1.0,
        min_distance: float = 0.0,
        move_factor: float = 0.5,
        soft_bias: float = 0.0,
        verbosity: int = 1,
        write_image: bool = False,
        write_parameters: bool = False,
    ):

    # Get the input image and parameters
    image = read_fits_file(fits_path)
    params = Namespace(
        filename=fits_path,
        out=out_path,
        par_out=param_out_path,
        alpha=alpha,
        bg_mean=bg_mean,
        bg_variance=bg_variance,
        gain=gain,
        min_distance=min_distance,
        move_factor=move_factor,
        soft_bias=soft_bias,
        verbosity=verbosity,
    )

    # Set the pixel type based on the type in the image
    params.d_type = c_float
    if np.issubdtype(image.dtype, np.float64):
        params.d_type = c_double
        mto.init_double_filtering(params)
    ct_classes.init_classes(params.d_type)
    
    # Pre-process the image
    processed_image = mto.preprocess_image(image, params, n=2)

    # Build a max tree
    mt = mto.build_max_tree(processed_image, params)

    # Filter the tree and find objects
    id_map, sig_ancs = mto.filter_tree(mt, processed_image, params)

    # Relabel objects for clearer visualisation
    id_map = mto.relabel_segments(id_map, shuffle_labels=False)

    # Generate output files
    if write_image:
        mto.generate_image(image, id_map, params)
    if write_parameters:
        mto.generate_parameters(image, id_map, sig_ancs, params)

    return {
        'image': image,
        'processed_image': processed_image,
        'id_map': id_map,
        'sig_ancs': sig_ancs,
        'params': params,
    }
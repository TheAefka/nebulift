import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from backend.pipeline import run_mto
from backend.classify import classify_objects, COMPACT, DIFFUSE, UNCLASSIFIED
from backend.stretch import apply_adaptive_stretch, asinh_stretch

class processingWorker(QThread):
    finished_success = Signal(np.ndarray) 
    finished_error = Signal(str)
    
    def __init__(self, fits_path, mto_params, class_params, stretch_params):
        super().__init__()
        self.fits_path = fits_path
        self.mto_params = mto_params
        self.class_params = class_params
        self.stretch_params = stretch_params
    
    def run(self):
        try:
            # MTOlib
            mto_results = run_mto(
                self.fits_path,
                alpha=self.mto_params.get('alpha', 1e-6),
                bg_mean=self.mto_params.get('bg_mean', None),
                bg_variance=self.mto_params.get('bg_variance', -1.0),
                gain=self.mto_params.get('gain', -1.0),
                min_distance=self.mto_params.get('min_distance', 0.0),
                move_factor=self.mto_params.get('move_factor', 0.5),
                soft_bias=self.mto_params.get('soft_bias', 0.0),
                verbosity=1
            )
            image = mto_results['image']

            image_parameters = pd.DataFrame(
                mto_results['image_parameters'][1:],
                columns=mto_results['image_parameters'][0]
            )

            # Classification
            r_threshold = self.class_params.get('r_fwhm_threshold', None)
            classified_parameters = classify_objects(image_parameters, r_fwhm_threshold=r_threshold)

            # Pixel-level class map
            id_map = mto_results['id_map'].astype(np.int64, copy=False)
            object_ids = classified_parameters['ID'].to_numpy(dtype=np.int64, copy=False)
            object_types = classified_parameters['source_type'].to_numpy(dtype=np.int8, copy=False)

            max_id = int(id_map.max())
            if object_ids.size:
                max_id = max(max_id, int(object_ids.max()))
            
            id_to_type_lut = np.full(max_id + 1, -1, dtype=np.int8)
            valid_object_rows = object_ids >= 0
            id_to_type_lut[object_ids[valid_object_rows]] = object_types[valid_object_rows]

            class_map = np.full(id_map.shape, -1, dtype=np.int8)
            valid_pixels = id_map >= 0
            class_map[valid_pixels] = id_to_type_lut[id_map[valid_pixels]]

            # Stretch
            c_stretch = self.stretch_params.get('compact', 100.0)
            d_stretch = self.stretch_params.get('diffuse', 10.0)
            bp = self.stretch_params.get('black_point', 0.001)

            stretched_image = apply_adaptive_stretch(
                image,
                class_map,
                lambda x: asinh_stretch(x, stretch_factor=c_stretch, black_point=bp),
                lambda x: asinh_stretch(x, stretch_factor=d_stretch, black_point=bp),
                black_point=bp,
                compact_label=COMPACT,
                diffuse_label=DIFFUSE
            )

            # Return np.ndarray to GUI
            self.finished_success.emit(stretched_image)

        except Exception as e:
            import traceback
            self.finished_error.emit(f"{str(e)}\n\n traceback.format_exec()")
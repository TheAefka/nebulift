import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from backend.pipeline import run_mto
from backend.classify import classify_objects, COMPACT, DIFFUSE
from backend.stretch import apply_adaptive_stretch

class processingWorker(QThread):
    finished_success = Signal(np.ndarray, np.ndarray, np.ndarray, dict) 
    finished_error = Signal(str)
    status_update = Signal(str)
    
    def __init__(self, class_params, stretch_params, fits_path = None, mto_params = None, mto_results = None, original_color_image = None):
        super().__init__()
        self.fits_path = fits_path
        self.mto_params = mto_params
        self.class_params = class_params
        self.stretch_params = stretch_params
        self.mto_results = mto_results
        self.original_color_image = original_color_image

    def run(self):
        try:
            if self.mto_results is None and self.fits_path is not None and self.mto_params is not None:
                # MTOlib
                self.status_update.emit("Building Max-Tree...")

                self.mto_results = run_mto(
                    fits_path=self.fits_path,
                    alpha=self.mto_params.get('alpha', 1e-6),
                    bg_mean=self.mto_params.get('bg_mean', None),
                    bg_variance=self.mto_params.get('bg_variance', -1.0),
                    gain=self.mto_params.get('gain', -1.0),
                    min_distance=self.mto_params.get('min_distance', 0.0),
                    move_factor=self.mto_params.get('move_factor', 0.5),
                    soft_bias=self.mto_params.get('soft_bias', 0.0),
                    verbosity=1
                )
            
            if self.original_color_image is not None:
                self.mto_results['original_color_image'] = self.original_color_image
            color_image = self.mto_results.get('original_color_image', None)
            stretch_input = color_image if color_image is not None else self.mto_results['image']
            image = self.mto_results['image']

            image_parameters = pd.DataFrame(
                self.mto_results['image_parameters'][1:],
                columns=self.mto_results['image_parameters'][0]
            )

            # Classification
            self.status_update.emit("Classifying Objects...")
            r_threshold = self.class_params.get('r_fwhm_threshold', None)
            a_b_threshold = self.class_params.get('a_b_threshold', None)
            classified_parameters = classify_objects(image_parameters, r_fwhm_threshold=r_threshold, a_b_threshold=a_b_threshold)

            # Pixel-level class map
            self.status_update.emit("Building Class Map for each pixel...")
            id_map = self.mto_results['id_map'].astype(np.int64, copy=False)
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
            self.status_update.emit("Applying Adaptative Stretching...")
            b_stretch = self.stretch_params.get('background', 5.0)
            c_stretch = self.stretch_params.get('compact', 100.0)
            d_stretch = self.stretch_params.get('diffuse', 10.0)
            bp = self.stretch_params.get('black_point', 0.001)

            sig_ancs = self.mto_results['sig_ancs']

            stretched_image = apply_adaptive_stretch(
                image=stretch_input,
                id_map=self.mto_results['id_map'],
                class_map=class_map,
                bg_stretch_factor=b_stretch,
                diffuse_stretch_factor=d_stretch,
                compact_stretch_factor=c_stretch,
                black_point=bp,
                compact_label=COMPACT,
                diffuse_label=DIFFUSE,
                mto_struct=self.mto_results['mto_struct'],
                id_to_type_lut=id_to_type_lut
            )

            if 'ID' in classified_parameters.columns:
                df_indexed = classified_parameters.set_index('ID')
                self.mto_results['object_parameters'] = df_indexed.to_dict(orient='index')
            else:
                self.mto_results['object_parameters'] = {}

            self.finished_success.emit(stretched_image, class_map, id_map, self.mto_results)

        except Exception as e:
            import traceback
            self.finished_error.emit(f"{str(e)}\n\n {traceback.format_exc()}")
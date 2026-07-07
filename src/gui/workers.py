import numpy as np
import time
import pandas as pd
from PySide6.QtCore import QThread, Signal

from backend.pipeline import run_mto
from backend.classify import (classify_objects_thresholds, classify_objects_gmm1,
                              classify_objects_gmm2, COMPACT, DIFFUSE)
from backend.stretch import apply_adaptive_stretch

class ProcessingWorker(QThread):
    finished_success = Signal(np.ndarray, np.ndarray, np.ndarray, dict) 
    finished_error = Signal(str)
    status_update = Signal(str)
    
    def __init__(
        self,
        class_params,
        stretch_params,
        fits_path = None,
        mto_params = None,
        mto_results = None,
        original_color_image = None,
        class_map = None,
        reclassify = True,
        lvq = None,
    ):
        super().__init__()
        self.fits_path = fits_path
        self.mto_params = mto_params
        self.class_params = class_params
        self.stretch_params = stretch_params
        self.mto_results = mto_results
        self.original_color_image = original_color_image
        self.class_map = class_map
        self.reclassify = reclassify
        self.lvq = lvq

    def run(self):
        t0 = time.time()
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
                print(f"Max-Tree built in {time.time() - t0:.2f} seconds.")

            if self.original_color_image is not None:
                self.mto_results['original_color_image'] = self.original_color_image
            color_image = self.mto_results.get('original_color_image', None)
            stretch_input = color_image if color_image is not None else self.mto_results['image']

            image_parameters = pd.DataFrame(
                self.mto_results['image_parameters'][1:],
                columns=self.mto_results['image_parameters'][0]
            )

            # Classification
            t1 = time.time()
            if self.lvq is not None:
                # LVQ path, apply correction then reclassify
                parameters_df = self.mto_results.get('object_parameters_df')
                if parameters_df is None:
                    op = self.mto_results.get('object_parameters', {})
                    parameters_df = pd.DataFrame.from_dict(op, orient='index')
                    parameters_df.index.name = 'ID'

                predicted = self.lvq.classify(parameters_df)

                id_map = self.mto_results['id_map'].astype(np.int64, copy=False)
                max_id = int(id_map.max())
                id_to_type_lut = np.full(max_id + 1, -1, dtype=np.int8)
                for obj_id, label in predicted.items():
                    obj_id = int(obj_id)
                    if 0 <= obj_id <= max_id:
                        id_to_type_lut[obj_id] = int(label)

                class_map = np.full(id_map.shape, -1, dtype=np.int8)
                valid_pixels = id_map >= 0
                class_map[valid_pixels] = id_to_type_lut[id_map[valid_pixels]]

                for obj_id, label in predicted.items():
                    obj_id = int(obj_id)
                    if obj_id in self.mto_results.get('object_parameters', {}):
                        self.mto_results['object_parameters'][obj_id]['source_type'] = int(label)

                self.mto_results['class_map'] = class_map
                print(f"LVQ classified {len(predicted)} objects: "
                      f"{(predicted == COMPACT).sum()} compact, "
                      f"{(predicted == DIFFUSE).sum()} diffuse")
                print(f"LVQ classification done in {time.time() - t1:.2f} seconds.")
            elif self.reclassify:
                self.status_update.emit("Classifying Objects...")
                classifier = (self.class_params or {}).get('classifier', 'gmm1')

                if classifier == 'threshold':
                    classified_parameters = classify_objects_thresholds(
                        image_parameters,
                        thresholds=self.class_params.get('thresholds', {})
                    )
                elif classifier == 'gmm2':
                    classified_parameters = classify_objects_gmm2(image_parameters)
                else:
                    classified_parameters = classify_objects_gmm1(image_parameters)
                
                class_counts = classified_parameters['source_type'].value_counts()
                print("Classification Summary:")
                for class_value, count in class_counts.items():
                    class_name = "Unclassified"
                    if class_value == DIFFUSE:
                        class_name = "Diffuse"
                    elif class_value == COMPACT:
                        class_name = "Compact"
                    print(f"  {class_name}: {count} objects")

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
                self.mto_results['class_map'] = class_map
                if 'object_parameters_df' in self.mto_results:
                    del self.mto_results['object_parameters_df']
                print(f"Classification and class map generation done in {time.time() - t1:.2f} seconds.")

            else:
                if self.class_map is None:
                    raise ValueError("Class map missing. Please reclassify objects.")
                class_map = self.class_map
                id_map = self.mto_results['id_map'].astype(np.int64, copy=False)
                
                id_to_type_lut = None
                if 'object_parameters' in self.mto_results:
                    max_id = int(id_map.max())
                    obj_ids = list(self.mto_results['object_parameters'].keys())
                    if obj_ids:
                        max_id = max(max_id, max(obj_ids))
                    id_to_type_lut = np.full(max_id + 1, -1, dtype=np.int8)
                    for obj_id, params in self.mto_results['object_parameters'].items():
                        if 'source_type' in params:
                            id_to_type_lut[obj_id] = params['source_type']
                print("Using existing classification and class map.")


            # Stretch
            t2 = time.time()
            self.status_update.emit("Applying Adaptative Stretching...")
            sp_stretch_factors = self.stretch_params['stretch_factors']
            sp_blackpoints     = self.stretch_params['blackpoints']
            sp_offsets         = self.stretch_params['offsets']
            sp_functions       = self.stretch_params['functions']

            stretched_image = apply_adaptive_stretch(
                image=stretch_input,
                id_map=self.mto_results['id_map'],
                class_map=class_map,
                stretch_factors=sp_stretch_factors,
                blackpoints=sp_blackpoints,
                offsets=sp_offsets,
                functions=sp_functions,
                mto_struct=self.mto_results['mto_struct'],
                id_to_type_lut=id_to_type_lut
            )
            print(f"Stretching done in {time.time() - t2:.2f} seconds.")

            if self.reclassify:
                if 'ID' in classified_parameters.columns:
                    type_series = classified_parameters.set_index('ID')['source_type']
                    full_df = image_parameters.set_index('ID').copy()
                    full_df['source_type'] = type_series
                    full_df.index = full_df.index.astype(int)
                    self.mto_results['object_parameters'] = full_df.to_dict(orient='index')
                else:
                    self.mto_results['object_parameters'] = {}
                self.mto_results['class_map'] = class_map

            self.finished_success.emit(stretched_image, class_map, id_map, self.mto_results)
            print(f"Total processing time: {time.time() - t0:.2f} seconds.\n")

        except Exception as e:
            import traceback
            self.finished_error.emit(f"{str(e)}\n\n {traceback.format_exc()}")
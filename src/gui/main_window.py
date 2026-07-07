import os
from astropy.io import fits
import numpy as np
import pandas as pd
from PIL import Image

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                               QSplitter, QMessageBox, QFileDialog)
from PySide6.QtGui import QIcon, QKeySequence, QImage

from backend.lvq_classifier import LVQClassifier
from gui.widgets.about_dialog import AboutDialog
from gui.widgets.settings_panel import SettingsPanel
from gui.widgets.image_panel import ImagePanel
from gui.widgets.object_info_panel import ObjectInfoPanel
from gui.widgets.line_plot_dialog import LinePlotDialog
from gui.workers import ProcessingWorker

# TODO: switch to Qt Resource Files (.qrc)?
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


SAVE_FORMATS = [
    ("FITS",  ["*.fits", "*.fit"], [".fits", ".fit"]),
    ("PNG",   ["*.png"],           [".png"]),
    ("JPEG",  ["*.jpg", "*.jpeg"], [".jpg",  ".jpeg"]),
    ("TIFF",  ["*.tif", "*.tiff"], [".tif",  ".tiff"]),
    ("All Files", ["*"],           [""]),
]



class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nebulift")
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))
        self.resize(1280, 720)
        self._lvq = LVQClassifier(features=['mu_max', 'mu_median', 'A/B', 'R_fwhm'])

        # Menu bar
        self._build_menu_bar()

        # Content
        container = QWidget()
        self.setCentralWidget(container)

        layout = QHBoxLayout(container)

        splitter = QSplitter()
        splitterLayout = QHBoxLayout(splitter)

        self.imagePanel = ImagePanel()
        self.leftPanel = SettingsPanel()
        self.rightPanel = ObjectInfoPanel()

        self.leftPanel.open_requested.connect(self.handle_open_image)
        
        layout.addWidget(splitter)
        splitterLayout.addWidget(self.leftPanel)
        splitterLayout.addWidget(self.imagePanel)
        splitterLayout.addWidget(self.rightPanel)
        splitter.setStretchFactor(1, 1)

        self.leftPanel.process_requested.connect(self.start_processing)
        self.cached_mto_results = None
        self.cached_stretched_image = None
        self.cached_class_map = None
        self.cached_classification_params = None
        self.cached_stretch_params = None
        self.leftPanel.stretch_requested.connect(self.update_stretch)
        self.leftPanel.classification_requested.connect(self.update_classification)
        self.leftPanel.overlay_toggled.connect(self.imagePanel._set_overlay)
        self.leftPanel.save_requested.connect(self.handle_save_image)

        self.imagePanel.pixel_selected_data.connect(self.rightPanel.update_info)
        self.rightPanel.reclassify_requested.connect(self.handle_reclassify)

        self.original_image = None

    def _aboutMessage(self):
        dialog = AboutDialog(self)
        dialog.exec()

    def _build_menu_bar(self):
        """
        Assemble all menu actions in the top menu bar.
        """
        menuBar = self.menuBar()
        
        fileMenu = menuBar.addMenu("File")
        editMenu = menuBar.addMenu("Edit")
        viewMenu = menuBar.addMenu("View")
        helpMenu = menuBar.addMenu("Help")

        openAction = fileMenu.addAction("Open")
        openAction.setShortcut(QKeySequence.StandardKey.Open)

        exitAction = fileMenu.addAction("Exit")
        exitAction.setShortcut(QKeySequence.StandardKey.Quit)

        undoAction = editMenu.addAction("Undo")
        undoAction.setShortcut(QKeySequence.StandardKey.Undo)
        undoAction.triggered.connect(self.handle_undo)

        redoAction = editMenu.addAction("Redo")
        redoAction.setShortcut(QKeySequence.StandardKey.Redo)
        redoAction.triggered.connect(self.handle_redo)

        editMenu.addAction("Reset LVQ").triggered.connect(self._lvq.reset)
        
        zoomInIcon = QIcon(os.path.join(BASE_DIR, "resources", "icons", "zoom-in.png"))
        zoomInAction = viewMenu.addAction(zoomInIcon, "Zoom In")
        zoomInAction.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        zoomInAction.triggered.connect(self._zoom_in)
        
        zoomOutIcon = QIcon(os.path.join(BASE_DIR, "resources", "icons", "zoom-out.png"))
        zoomOutAction = viewMenu.addAction(zoomOutIcon, "Zoom Out")
        zoomOutAction.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoomOutAction.triggered.connect(self._zoom_out)
        
        resetZoomIcon = QIcon(os.path.join(BASE_DIR, "resources", "icons", "reset-zoom.png"))
        resetZoomAction = viewMenu.addAction(resetZoomIcon, "Reset Zoom")
        resetZoomAction.setShortcut(QKeySequence("Ctrl+0"))
        resetZoomAction.triggered.connect(self._reset_zoom)
        
        viewMenu.addSeparator()
        linePlotAction = viewMenu.addAction("Line Plot...")
        linePlotAction.setShortcut(QKeySequence("Ctrl+L"))
        linePlotAction.triggered.connect(self._show_line_plot_dialog)

        openAction.triggered.connect(self.handle_open_image)
        exitAction.triggered.connect(self.close)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(self._aboutMessage)
        aboutAction.setShortcut(QKeySequence("F1"))

    def handle_open_image(self):
        """
        Handler for opening a new FITS image. Resets app state.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)",
        )
        if file_path:
            self.current_fits_path = file_path
            self.original_image = self._load_original_image(file_path)
            self.imagePanel.load_file(file_path)
            self.imagePanel.fit_to_view()
            self.leftPanel.update_fits_display(file_path)

            # Reset state
            self.cached_mto_results = None
            self.cached_stretched_image = None
            self.cached_class_map = None
            self.cached_classification_params = None
            self.cached_stretch_params = None
            self.leftPanel.apply_classification_btn.setEnabled(False)
            self.leftPanel.apply_stretch_btn.setEnabled(False)
            self._lvq.reset_on_image_change()

    def handle_reclassify(self, object_id: int, new_class: int):
        """
        Handle a user reclassification of an object. Update LVQ and re-stretch if ready.
        """
        if self.cached_mto_results is None:
            return

        using_lvq = self.leftPanel.selected_classifier() == 'lvq'

        self.leftPanel.setEnabled(False)
        df = self.cached_mto_results.get('object_parameters_df')

        if using_lvq and df is not None:
            self._lvq.add_label(object_id, new_class, df)
            print(f"LVQ labels: {self._lvq.labelled_ids}, ready: {self._lvq.is_ready()}")

        if using_lvq and self._lvq.is_ready():
            # TODO: Update LVQ hint label with stats


            if hasattr(self, 'stretch_worker') and self.stretch_worker.isRunning():
                self.stretch_worker.wait()

            self.stretch_worker = ProcessingWorker(
                mto_results=self.cached_mto_results,
                class_params=self.cached_classification_params,
                stretch_params=self.cached_stretch_params,
                class_map=self.cached_class_map,
                reclassify=False,
                original_color_image=self.original_image,
                lvq=self._lvq,
            )
            self.stretch_worker.status_update.connect(self.imagePanel.coord_label.setText)
            self.stretch_worker.finished_error.connect(self.on_processing_error)
            self.stretch_worker.finished_success.connect(self._on_stretch_finished)
            self.stretch_worker.start()
        else:
            # Non-LVQ: single object direct override then re-stretch
            id_map = self.cached_mto_results['id_map'].astype(np.int64, copy=False)
            self.cached_class_map[id_map == object_id] = new_class
            self.cached_mto_results['class_map'] = self.cached_class_map
            if object_id in self.cached_mto_results.get('object_parameters', {}):
                self.cached_mto_results['object_parameters'][object_id]['source_type'] = new_class
            self.update_stretch(self.cached_stretch_params)


    def handle_save_image(self):
        """
        On Save Image request, open a file dialog and save the currently
        displayed stretched image to the selected path in the chosen format.
        """

        if self.cached_stretched_image is None:
            QMessageBox.information(
                self,
                "No image to save",
                "Please process an image before saving.",
            )
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            ";;".join([f"{fmt[0]} ({' '.join(fmt[1])})" for fmt in SAVE_FORMATS])
        )
        if not file_path:
            return

        if not os.path.splitext(file_path)[1]:
            try:
                file_path += SAVE_FORMATS[[f[0] for f in SAVE_FORMATS].index(selected_filter.split()[0])][2][0]
            except ValueError:
                file_path += ".fits"
        
        extension = os.path.splitext(file_path)[1].lower()

        try:
            data = np.flipud(self.cached_stretched_image)

            if extension in ['.fits', '.fit']:
                hdu = fits.PrimaryHDU(np.moveaxis(self.cached_stretched_image, -1, 0) if data.ndim == 3 else data)
                hdu.writeto(file_path, overwrite=True)
            else:
                uint8_data = (np.clip(data, 0.0, 1.0)*255).astype(np.uint8)
                uint8_data = np.ascontiguousarray(uint8_data)

                Image.fromarray(uint8_data).save(file_path)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Image",
                f"An error occurred while saving the image:\n{str(e)}",
            )

    def _load_original_image(self, file_path):
        try:
            with fits.open(file_path) as f:
                data = f[0].data.astype(np.float32)
            # Move channel axis to the end
            if data.ndim == 3 and data.shape[0] in (3, 4):
                data = np.moveaxis(data, 0, -1)
            return data
        except Exception:
            return None

    def _zoom_in(self):
        self.imagePanel.zoom_in()

    def _zoom_out(self):
        self.imagePanel.zoom_out()

    def _reset_zoom(self):
        self.imagePanel.fit_to_view()

    def start_processing(self, fits_path, mto_params, class_params, stretch_params):
        self.leftPanel.setEnabled(False)
        self.cached_classification_params = class_params
        self.cached_stretch_params = stretch_params

        self.worker = ProcessingWorker(fits_path=fits_path, mto_params=mto_params, class_params=class_params, stretch_params=stretch_params, original_color_image=self.original_image)
        self.worker.status_update.connect(self.imagePanel.coord_label.setText)
        self.worker.finished_success.connect(self.on_processing_success)
        self.worker.finished_error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_success(self, stretched_image_array, class_map, id_map, mto_results):
        self.leftPanel.setEnabled(True)
        self.leftPanel.apply_classification_btn.setEnabled(True)
        self.leftPanel.apply_stretch_btn.setEnabled(True)
        self.leftPanel.save_btn.setEnabled(True)
        self.cached_mto_results = mto_results
        self._rebuild_parameters_df()
        self.cached_stretched_image = stretched_image_array
        self.cached_class_map = class_map
        self.cached_mto_results['class_map'] = class_map
        self.imagePanel.load_numpy_array(
            stretched_image_array,
            class_map=class_map,
            id_map=id_map,
            original_image=self.original_image,
            mto_struct=mto_results['mto_struct'],
            object_parameters=mto_results.get('object_parameters')
        )
        self.imagePanel.fit_to_view()
        self.worker.deleteLater()

    def on_processing_error(self, error_msg):
        self.leftPanel.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", error_msg)
        self.imagePanel.coord_label.setText("Error during processing.")
        self.worker.deleteLater()

    def update_classification(self, classif, stretch):
        if self.cached_mto_results is None:
            return

        self.leftPanel.setEnabled(False)

        self.cached_classification_params = classif
        self.cached_stretch_params = stretch

        if hasattr(self, 'stretch_worker') and self.stretch_worker.isRunning():
            self.stretch_worker.wait()

        classifier = classif.get('classifier', 'gmm1')

        if classifier == 'lvq':
            df = (self.cached_mto_results or {}).get('object_parameters_df')
            if df is not None and not self._lvq.labelled_ids and self._lvq.persistent_labels:
                self._lvq.refit_from_persistent(df)

            if self._lvq.is_ready(): # Enough labels
                self.stretch_worker = ProcessingWorker(
                    mto_results=self.cached_mto_results,
                    class_params=classif,
                    stretch_params=stretch,
                    class_map=self.cached_class_map,
                    reclassify=False,
                    original_color_image=self.original_image,
                    lvq=self._lvq,
                )
            else:
                # Not enough labels, reuse existing class_map and re-stretch
                self.stretch_worker = ProcessingWorker(
                    mto_results=self.cached_mto_results,
                    class_params=classif,
                    stretch_params=stretch,
                    class_map=self.cached_class_map,
                    reclassify=False,
                    original_color_image=self.original_image,
                )
        else:
            # Any GMM/threshold: reset LVQ and reclassify fresh
            self._lvq.reset()
            self.stretch_worker = ProcessingWorker(
                mto_results=self.cached_mto_results,
                class_params=classif,
                stretch_params=stretch,
            )

        self.stretch_worker.status_update.connect(self.imagePanel.coord_label.setText)
        self.stretch_worker.finished_error.connect(self.on_processing_error)
        self.stretch_worker.finished_success.connect(self._on_stretch_finished)
        self.stretch_worker.start()

    def update_stretch(self, stretch):
        if self.cached_mto_results is None:
            return
        
        self.leftPanel.setEnabled(False)
        self.cached_stretch_params = stretch

        if hasattr(self, 'stretch_worker') and self.stretch_worker.isRunning():
            self.stretch_worker.wait() # Wait for finished
    
        self.stretch_worker = ProcessingWorker(
            mto_results=self.cached_mto_results,
            class_params=self.cached_classification_params,
            stretch_params=self.cached_stretch_params,
            class_map=self.cached_class_map,
            reclassify=False,
            original_color_image=self.original_image
        )
        self.stretch_worker.finished_error.connect(self.on_processing_error)
        self.stretch_worker.finished_success.connect(self._on_stretch_finished)
        self.stretch_worker.start()
    
    def _on_stretch_finished(self, new_image, class_map, id_map, mto_results):
        self.cached_stretched_image = new_image
        self.cached_mto_results = mto_results
        self._rebuild_parameters_df()
        self.cached_class_map = class_map
        self.cached_mto_results['class_map'] = class_map
        self.imagePanel.load_numpy_array(
            new_image,
            class_map=class_map,
            id_map=id_map,
            preserve_zoom=True,
            original_image=self.original_image,
            mto_struct=mto_results.get('mto_struct'),
            object_parameters=mto_results.get('object_parameters')
        )
        self.leftPanel.setEnabled(True)
    
    def _show_line_plot_dialog(self):
        if self.cached_stretched_image is None:
            QMessageBox.information(
                self,
                "No data",
                "Please open and process an image first.",
            )
            return

        stretched = self.cached_stretched_image
        stretched_display = np.flipud(stretched)

        view = self.imagePanel.view
        top_left = view.mapToScene(view.viewport().rect().topLeft())
        bottom_right = view.mapToScene(view.viewport().rect().bottomRight())

        x0 = max(0, int(np.floor(top_left.x())))
        x1 = min(stretched_display.shape[1], int(np.ceil(bottom_right.x())))
        y0 = max(0, int(np.floor(top_left.y())))
        y1 = min(stretched_display.shape[0], int(np.ceil(bottom_right.y())))

        selected_pixel = self.imagePanel.pixel_selected
        if selected_pixel is None:
            return
        x, y = selected_pixel

        original_display = np.flipud(self.original_image) if self.original_image is not None else None
        class_map_display = np.flipud(self.cached_class_map) if self.cached_class_map is not None else None

        dialog = LinePlotDialog(
            original_display,
            stretched_display,
            x0,
            x1,
            y0,
            y1,
            x,
            y,
            class_map=class_map_display,
            parent=self,
        )
        if dialog.exec() == LinePlotDialog.DialogCode.Accepted:
            dialog.show_plot()



    def _rebuild_parameters_df(self):
        """Build a DataFrame from object_parameters and cache it for LVQ."""
        op = self.cached_mto_results.get('object_parameters')
        if op:
            df = pd.DataFrame.from_dict(op, orient='index')
            df.index = df.index.astype(int)
            df.index.name = 'ID'
            for col in self._lvq.features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            self.cached_mto_results['object_parameters_df'] = df
        else:
            self.cached_mto_results['object_parameters_df'] = None


    def _apply_lvq_history_state(self):
        """Helper to apply LVQ changes cleanly after undo/redo."""
        self.leftPanel.setEnabled(False)

        if hasattr(self, 'stretch_worker') and self.stretch_worker.isRunning():
            self.stretch_worker.wait()

        if self._lvq.is_ready():
            # Re-apply LVQ model over image
            self.stretch_worker = ProcessingWorker(
                mto_results=self.cached_mto_results,
                class_params=self.cached_classification_params,
                stretch_params=self.cached_stretch_params,
                class_map=self.cached_class_map,
                reclassify=False,
                original_color_image=self.original_image,
                lvq=self._lvq,
            )
        else:
            # If undoing drops below the required label count, revert to base classifier
            self.stretch_worker = ProcessingWorker(
                mto_results=self.cached_mto_results,
                class_params=self.cached_classification_params,
                stretch_params=self.cached_stretch_params,
            )

        self.stretch_worker.status_update.connect(self.imagePanel.coord_label.setText)
        self.stretch_worker.finished_error.connect(self.on_processing_error)
        self.stretch_worker.finished_success.connect(self._on_stretch_finished)
        self.stretch_worker.start()

    def handle_undo(self):
        df = self.cached_mto_results.get('object_parameters_df', None)
        if df is None:
            return
        self._lvq.undo(df)
        self._apply_lvq_history_state()
    
    def handle_redo(self):
        df = self.cached_mto_results.get('object_parameters_df', None)
        if df is None:
            return
        self._lvq.redo(df)
        self._apply_lvq_history_state()
import os
from astropy.io import fits
import numpy as np

from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                               QSplitter, QMessageBox, QFileDialog)
from PySide6.QtGui import QIcon, QKeySequence

from gui.widgets.about_dialog import AboutDialog
from gui.widgets.settings_panel import SettingsPanel
from gui.widgets.image_panel import ImagePanel
from gui.widgets.object_info_panel import ObjectInfoPanel
from gui.workers import processingWorker

# TODO: switch to Qt Resource Files (.qrc)?
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nebulift")
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))
        self.resize(1280, 720)

        # Menu bar
        menuBar = self.menuBar()
        
        fileMenu = menuBar.addMenu("File")
        editMenu = menuBar.addMenu("Edit")
        viewMenu = menuBar.addMenu("View")
        helpMenu = menuBar.addMenu("Help")

        openAction = fileMenu.addAction("Open")
        openAction.setShortcut(QKeySequence.StandardKey.Open)
        exitAction = fileMenu.addAction("Exit")
        exitAction.setShortcut(QKeySequence.StandardKey.Quit)

        editMenu.addAction("Undo").setShortcut(QKeySequence.StandardKey.Undo)
        editMenu.addAction("Redo").setShortcut(QKeySequence.StandardKey.Redo)
        
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

        openAction.triggered.connect(self.handle_open_image)
        exitAction.triggered.connect(self.close)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(self._aboutMessage)
        aboutAction.setShortcut(QKeySequence("F1"))

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
        self.leftPanel.parameters_changed.connect(self.update_stretch)
        self.leftPanel.overlay_toggled.connect(self.imagePanel._set_overlay)

        self.imagePanel.pixel_selected_data.connect(self.rightPanel.update_info)

        self.original_image = None

    def _aboutMessage(self):
        dialog = AboutDialog(self)
        dialog.exec()



    def handle_open_image(self):
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

        self.worker = processingWorker(fits_path=fits_path, mto_params=mto_params, class_params=class_params, stretch_params=stretch_params, original_color_image=self.original_image)
        self.worker.status_update.connect(self.imagePanel.coord_label.setText)
        self.worker.finished_success.connect(self.on_processing_success)
        self.worker.finished_error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_success(self, stretched_image_array, class_map, id_map, mto_results):
        self.leftPanel.setEnabled(True)
        self.cached_mto_results = mto_results
        self.cached_stretched_image = stretched_image_array
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

    def update_stretch(self, classif, stretch):
        if self.cached_mto_results is None:
            return
        
        self.leftPanel.setEnabled(False)

        if hasattr(self, 'stretch_worker') and self.stretch_worker.isRunning():
            self.stretch_worker.wait() # Wait for finished

        self.stretch_worker = processingWorker(mto_results=self.cached_mto_results, class_params=classif, stretch_params=stretch)
        self.stretch_worker.finished_success.connect(self._on_stretch_finished)
        self.stretch_worker.start()
    
    def _on_stretch_finished(self, new_image, class_map, id_map, mto_results):
        self.cached_stretched_image = new_image
        self.cached_mto_results = mto_results
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
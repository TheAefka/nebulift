import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                               QSlider, QProgressBar, QComboBox, QCheckBox,
                               QSplitter, QMessageBox, QDialog, QDialogButtonBox,
                               QLayout, QFileDialog)
from PySide6.QtGui import QIcon, QPixmap, QKeySequence
from PySide6.QtCore import Qt

from gui.widgets.about_dialog import AboutDialog
from gui.widgets.settings_panel import SettingsPanel
from gui.widgets.image_panel import ImagePanel
from gui.workers import processingWorker

# TODO: switch to Qt Resource Files (.qrc)
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
        exitAction = fileMenu.addAction("Exit")
        
        zoomInAction = viewMenu.addAction("Zoom In")
        zoomInAction.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        zoomInAction.triggered.connect(self._zoom_in)
        zoomOutAction = viewMenu.addAction("Zoom Out")
        zoomOutAction.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoomOutAction.triggered.connect(self._zoom_out)

        openAction.triggered.connect(self.handle_open_image)
        exitAction.triggered.connect(self.close)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(self._aboutMessage)

        # Content
        container = QWidget()
        self.setCentralWidget(container)

        layout = QHBoxLayout(container)

        splitter = QSplitter()
        splitterLayout = QHBoxLayout(splitter)

        self.imagePanel = ImagePanel()
        self.leftPanel = SettingsPanel()

        self.leftPanel.open_requested.connect(self.handle_open_image)
        
        layout.addWidget(splitter)
        splitterLayout.addWidget(self.leftPanel)
        splitterLayout.addWidget(self.imagePanel)
        splitter.setStretchFactor(1, 1)

        self.leftPanel.process_requested.connect(self.start_processing)
        self.cached_mto_results = None
        self.leftPanel.parameters_changed.connect(self.update_stretch)
        self.leftPanel.overlay_toggled.connect(self.imagePanel._set_overlay)


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

            self.imagePanel.load_file(file_path)
            self.imagePanel.fit_to_view()
            self.leftPanel.update_fits_display(file_path)

    def _zoom_in(self):
        self.imagePanel.zoom_in()

    def _zoom_out(self):
        self.imagePanel.zoom_out()

    
    def start_processing(self, fits_path, mto_params, class_params, stretch_params):
        self.leftPanel.setEnabled(False)

        self.worker = processingWorker(fits_path=fits_path, mto_params=mto_params, class_params=class_params, stretch_params=stretch_params)
        self.worker.status_update.connect(self.imagePanel.image_label.setText)
        self.worker.finished_success.connect(self.on_processing_success)
        self.worker.finished_error.connect(self.on_processing_error)
        self.worker.start()

    def on_processing_success(self, stretched_image_array, class_map, id_map, mto_results):
        self.leftPanel.setEnabled(True)
        self.cached_mto_results = mto_results
        self.imagePanel.load_numpy_array(stretched_image_array, class_map=class_map, id_map=id_map, sig_ancs=mto_results['sig_ancs'],)
        self.imagePanel.fit_to_view()
        self.worker.deleteLater()

    def on_processing_error(self, error_msg):
        self.leftPanel.setEnabled(True)
        QMessageBox.critical(self, "Pipeline Error", error_msg)
        self.imagePanel.image_label.setText("Error during processing.")
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
    
    def _on_stretch_finished(self, new_image, class_map, id_map, sig_ancs):
        self.imagePanel.load_numpy_array(new_image, class_map=class_map, id_map=id_map, sig_ancs=sig_ancs, preserve_zoom=True)
        self.leftPanel.setEnabled(True)
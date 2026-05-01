import os
from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QComboBox,
                               QSlider, QSpinBox, QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage


class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("No Image Loaded")
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area)

    def load_file(self, file_path):
        pixmap = QPixmap(self._convert_fits_to_qimage(file_path))
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setText("Failed to load image.")
    
    def _convert_fits_to_qimage(self, fits_path):
        with fits.open(fits_path) as fits_image:
            data = fits_image[0].data

            # Normalize to 0-255
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

            height, width = normalized.shape
            return QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
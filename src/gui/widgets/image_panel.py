import os
from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QComboBox,
                               QSlider, QSpinBox, QScrollArea)
from PySide6.QtCore import Qt, Slot, QEvent
from PySide6.QtGui import QPixmap, QImage


class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self._pan_start = None

        self.layout = QVBoxLayout(self)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel("No Image Loaded")
        self.image_label.setScaledContents(True)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area)

        self.scroll_area.installEventFilter(self)
        self.scroll_area.viewport().installEventFilter(self)

    def load_file(self, file_path):
        pixmap = QPixmap(self._convert_fits_to_qimage(file_path))
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap)
            self.scale_factor = 1.0
            self.image_label.adjustSize()
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
    
    def load_numpy_array(self, image_data: np.ndarray, preserve_zoom: bool = False):
        scaled_data = (image_data * 255.0).clip(0, 255).astype(np.uint8)
        height, width = scaled_data.shape
        self._current_image_data = scaled_data
        qimage = QImage(self._current_image_data.data, width, height, width, QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(qimage))

        if preserve_zoom:
            new_size = self.image_label.pixmap().size() * self.scale_factor
            self.image_label.resize(new_size)
        else:
            self.scale_factor = 1.0
            self.image_label.adjustSize()
    
    def _scale_image(self, factor):
        if self.image_label.pixmap():
            self.scale_factor *= factor
            new_size = self.image_label.pixmap().size() * self.scale_factor
            self.image_label.resize(new_size)

    @Slot()
    def zoom_in(self):
        self._scale_image(1.25)

    @Slot()
    def zoom_out(self):
        self._scale_image(0.8)

    def eventFilter(self, source, event):
        # Mouse panning with middle click+drag
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MiddleButton:
                self._pan_start = event.globalPosition().toPoint()
                self.scroll_area.viewport().setCursor(Qt.ClosedHandCursor)
                return True
        
        elif event.type() == QEvent.Type.MouseMove:
            if self._pan_start is not None:
                delta = event.globalPosition().toPoint() - self._pan_start
                self._pan_start = event.globalPosition().toPoint()
                self.scroll_area.horizontalScrollBar().setValue(
                    self.scroll_area.horizontalScrollBar().value() - delta.x()
                )
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() - delta.y()
                )
                return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MiddleButton:
                self._pan_start = None
                self.scroll_area.viewport().setCursor(Qt.ArrowCursor)
                return True

        # Zoom in/out with ctrl+scroll
        if event.type() == QEvent.Type.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                mouse_pos = event.position()
                old_pos = self.image_label.mapFrom(self.scroll_area.viewport(), mouse_pos.toPoint())
                factor = 1.1 if event.angleDelta().y() > 0 else 0.9
                self._scale_image(factor)

                new_pos = old_pos * factor
                delta = new_pos - old_pos

                self.scroll_area.horizontalScrollBar().setValue(
                    self.scroll_area.horizontalScrollBar().value() + delta.x()
                )
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() + delta.y()
                )

                event.accept()
                return True  # == Event handled
        
        return super().eventFilter(source, event)

    def fit_to_view(self):
        if not self.image_label.pixmap():
            return
        
        view_size = self.size()
        pix_size = self.image_label.pixmap().size()

        factor = min(view_size.width() / pix_size.width(), 
                     view_size.height() / pix_size.height())
        
        self.set_zoom(factor * 0.95)
    
    def set_zoom(self, factor):
        if self.image_label.pixmap():
            self.scale_factor = factor
            new_size = self.image_label.pixmap().size() * self.scale_factor
            self.image_label.resize(new_size)
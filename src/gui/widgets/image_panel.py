import os
from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea)
from PySide6.QtCore import Qt, Slot, QEvent
from PySide6.QtGui import QPixmap, QImage
from backend.classify import DIFFUSE, COMPACT

class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self._pan_start = None
        self._current_image_data = None
        self._current_class_map = None
        self._current_id_map = None
        self._overlay_toggled = False

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
    
    def load_numpy_array(self, image_data: np.ndarray, class_map: np.ndarray = None, id_map: np.ndarray = None, preserve_zoom: bool = False):
        self._current_image_data = (image_data * 255.0).clip(0, 255).astype(np.uint8)
        if class_map is not None:
            self._current_class_map = class_map
        if id_map is not None:
            self._current_id_map = id_map
        self._render_image(preserve_zoom=preserve_zoom)

    def _render_image(self, preserve_zoom = False):
        if self._current_image_data is None:
            return

        if self._overlay_toggled and self._current_class_map is not None:
            qimage = self._build_classif_overlay()
        else:
            height, width = self._current_image_data.shape
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

    def _build_classif_overlay(self) -> QImage:
        grey = self._current_image_data
        height, width = grey.shape
        rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)
        alpha = 0.3
        for class_value in [DIFFUSE, COMPACT]:
            mask = self._current_class_map == class_value
            if mask.any():
                r = (class_value == COMPACT) * 255 * alpha + grey[mask] * (1 - alpha)
                g = (class_value == DIFFUSE) * 255 * alpha + grey[mask] * (1 - alpha)
                b = grey[mask]
                rgb[mask] = np.stack([r, g, b], axis=1)
                # rgb[mask] = (np.array([(class_value==COMPACT)*255, (class_value==DIFFUSE)*255, 0], dtype=np.float32))
        image_array = np.ascontiguousarray(rgb.clip(0, 255).astype(np.uint8))
        return QImage(image_array.data, width, height, 3 * width, QImage.Format_RGB888)
    
    def _set_overlay(self, enabled):
        self._overlay_toggled = enabled
        self._render_image(preserve_zoom=True)

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
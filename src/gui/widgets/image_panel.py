import os
from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PySide6.QtCore import Qt, Slot, QEvent
from PySide6.QtGui import QPainter, QPixmap, QImage, QCursor, QColor
from backend.classify import DIFFUSE, COMPACT

class ImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self._pan_start = None
        self._current_image_data = None
        self._current_class_map = None
        self._current_id_map = None
        self._current_mto_struct = None
        self._overlay_toggled = False
        self._original_image = None
        self._current_sig_ancs = None
        self._pixel_selected = None

        self.layout = QVBoxLayout(self)

        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        self.view.setRenderHint(QPainter.Antialiasing, False)
        self.view.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.view.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.FastTransformation)
        self.scene.addItem(self.pixmap_item)

        # Coordinate display
        self.coord_label = QLabel("")
        self.layout.addWidget(self.coord_label)

        self.view.setMouseTracking(True)
        self.layout.addWidget(self.view)
        self.view.viewport().installEventFilter(self)
        

    def load_file(self, file_path):
        qimg = self._convert_fits_to_qimage(file_path)
        if not qimg.isNull():
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
            self.scale_factor = 1.0
        else:
            self.coord_label.setText("Failed to load image.")
    
    def _convert_fits_to_qimage(self, fits_path):
        with fits.open(fits_path) as fits_image:
            data = fits_image[0].data

            # Normalize to 0-255
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

            height, width = normalized.shape
            return QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
    
    def load_numpy_array(self, image_data: np.ndarray, class_map: np.ndarray = None, id_map: np.ndarray = None, sig_ancs: np.ndarray = None, preserve_zoom: bool = False, original_image: np.ndarray = None, mto_struct = None):
        self._current_image_data = (image_data * 255.0).clip(0, 255).astype(np.uint8)
        if class_map is not None:
            self._current_class_map = class_map
        if id_map is not None:
            self._current_id_map = id_map
        if sig_ancs is not None:
            self._current_sig_ancs = sig_ancs
        if mto_struct is not None:
            self._current_mto_struct = mto_struct
        if original_image is not None:
            self._original_image = np.asarray(original_image, dtype=np.float32)

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
            new_size = self.pixmap_item.pixmap().size() * self.scale_factor
            self.pixmap_item.setPixmap(self.pixmap_item.pixmap().scaled(new_size))
        else:
            self.scale_factor = 1.0
            self.pixmap_item.setPixmap(self.pixmap_item.pixmap().scaled(self.scale_factor))

    
    def _scale_image(self, factor):
        if self.pixmap_item.pixmap():
            self.scale_factor *= factor
            self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.view.scale(factor, factor)

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
                self.view.viewport().setCursor(Qt.ClosedHandCursor)
                return True
        
        elif event.type() == QEvent.Type.MouseMove:
            # Update coordinate display
            scene_pos = self.view.mapToScene(event.pos())
            img_x = int(scene_pos.x())
            img_y = int(scene_pos.y())
            pixmap = self.pixmap_item.pixmap()
            
            if pixmap:
                if (
                    0 <= img_x < pixmap.width()
                    and 0 <= img_y < pixmap.height()
                ):
                    self.coord_label.setText(f"x: {img_x}, y: {img_y}")
                else:
                    self.coord_label.setText("")

            if self._pan_start is not None:
                delta = event.globalPosition().toPoint() - self._pan_start
                self._pan_start = event.globalPosition().toPoint()
                self.view.horizontalScrollBar().setValue(
                    self.view.horizontalScrollBar().value() - delta.x()
                )
                self.view.verticalScrollBar().setValue(
                    self.view.verticalScrollBar().value() - delta.y()
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
                factor = 1.1 if event.angleDelta().y() > 0 else 0.9
                self._scale_image(factor)
                event.accept()
                return True

        # Trackpad pinch zoom
        if event.type() == QEvent.Type.NativeGesture:
            if event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                factor = 1.0 + event.value()
                self._scale_image(factor)
                return True

        return super().eventFilter(source, event)

    def fit_to_view(self):
        if not self.pixmap_item.pixmap():
            return
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def set_zoom(self, factor):
        self.view.resetTransform()
        self.scale_factor = 1.0
        self._scale_image(factor)


from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PySide6.QtCore import Qt, Signal, Slot, QEvent
from PySide6.QtGui import QPainter, QPixmap, QImage, QColor
from backend.classify import DIFFUSE, COMPACT

class ImagePanel(QWidget):
    pixel_selected_data = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.0
        self._pan_start = None
        self._current_image_data = None
        self._current_image_data_rgb = None
        self._current_class_map = None
        self._current_id_map = None
        self._current_mto_struct = None
        self._overlay_toggled = False
        self._original_image = None
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
            self._set_pixmap(QPixmap.fromImage(qimg), preserve_zoom=False)
        else:
            self.coord_label.setText("Failed to load image.")
    
    def _convert_fits_to_qimage(self, fits_path):
        with fits.open(fits_path) as fits_image:
            data = fits_image[0].data
            if data.ndim == 3 and data.shape[0] in [3, 4]:
                data = np.moveaxis(data, 0, -1)
            data = np.flipud(data)
            # Normalize to 0-255
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if data_max > data_min:
                normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)
            normalized = np.ascontiguousarray(normalized)
            if normalized.ndim == 2:
                height, width = normalized.shape
                return QImage(normalized.data, width, height, width, QImage.Format_Grayscale8)
            elif normalized.ndim == 3:
                height, width, channels = normalized.shape
                if channels == 3:
                    return QImage(normalized.data, width, height, channels * width, QImage.Format_RGB888)
                elif channels == 4:
                    return QImage(normalized.data, width, height, channels * width, QImage.Format_ARGB32)
        return QImage()
    
    def load_numpy_array(self, image_data: np.ndarray, class_map: np.ndarray = None, id_map: np.ndarray = None, preserve_zoom: bool = False, original_image: np.ndarray = None, mto_struct = None, object_parameters: dict = None):
        data = np.asarray(image_data, dtype=np.float32)
        if data.ndim == 3:
            self._current_image_data_rgb = np.flipud(
                (np.clip(data, 0, 1) * 255).astype(np.uint8)
            )
            self._current_image_data = np.flipud(data.mean(axis=2))
        else:
            self._current_image_data = np.flipud(data)
            self._current_image_data_rgb = None

        self._current_class_map = np.flipud(class_map) if class_map is not None else None
        self._current_id_map = np.flipud(id_map) if id_map is not None else None
        self._current_mto_struct = mto_struct if mto_struct is not None else None
        self._object_parameters = object_parameters if object_parameters is not None else {}

        self._original_image = np.flipud(np.asarray(original_image, dtype=np.float32)) if original_image is not None else None

        self._render_image(preserve_zoom=preserve_zoom)


    def get_direct_parent(self, obj_id: int, id_map_flat: np.ndarray, nodes, img_data) -> tuple:
        parent_idx = int(nodes[obj_id].parent)
        parent_val = float(img_data[parent_idx])
        parent_obj = int(id_map_flat[parent_idx])   # -1 if background
        return parent_val, parent_obj, parent_idx


    def _render_image(self, preserve_zoom = False):
        if self._current_image_data is None:
            return

        if self._overlay_toggled and self._current_class_map is not None:
            qimage = self._build_classif_overlay()
        else:
            qimage = self._img_to_rgb()
        overlay_array = self._build_selection_overlay_array()
        if overlay_array is not None:
            self.last_qimage_array = np.ascontiguousarray(overlay_array)
            h, w, _ = overlay_array.shape
            overlay_qimage = QImage(self.last_qimage_array.data, w, h, 4 * w, QImage.Format_RGBA8888)

            result = QImage(qimage.size(), QImage.Format_RGB888)
            result.fill(QColor(0, 0, 0))  # start blank
            painter = QPainter(result)
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            painter.drawImage(0, 0, qimage)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            painter.drawImage(0, 0, overlay_qimage)
            painter.end()
            out_qimg = result
        else:
            out_qimg = qimage
        self._set_pixmap(QPixmap.fromImage(out_qimg), preserve_zoom=preserve_zoom)

    def _to_uint8(self, data):
        data = np.asarray(data, dtype=np.float32)
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_max <= data_min:
            return np.zeros_like(data, dtype=np.uint8)
        
        normalized = (data - data_min) / (data_max - data_min)
        return (normalized * 255).astype(np.uint8)


    def _img_to_rgb(self):
        if self._current_image_data_rgb is not None:
            array = self._current_image_data_rgb
        else:
            grey = self._to_uint8(self._current_image_data)
            array = np.stack([grey, grey, grey], axis=2).astype(np.uint8)
        height, width = array.shape[:2]
        array = np.ascontiguousarray(array)
        self._last_qimage_array = array
        return QImage(self._last_qimage_array.data, width, height, 3 * width, QImage.Format_RGB888)

    def _scale_image(self, factor):
        if self.pixmap_item.pixmap():
            self.scale_factor *= factor
            self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.view.scale(factor, factor)

    def _set_pixmap(self, pixmap: QPixmap, preserve_zoom: bool):
        if not preserve_zoom:
            self.view.resetTransform()
            self.scale_factor = 1.0

        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def _build_classif_overlay(self) -> QImage:
        grey = self._to_uint8(self._current_image_data)
        height, width = grey.shape
        if self._current_image_data_rgb is not None:
            rgb = self._current_image_data_rgb.astype(np.float32)
        else:
            rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)

        alpha = 0.3
        for class_value, tint in [(COMPACT, np.array([255, 0, 0])), (DIFFUSE, np.array([0, 255, 0]))]:
            mask = self._current_class_map == class_value
            if mask.any():
                rgb[mask] = rgb[mask] * (1 - alpha) + tint * alpha

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
            scene_pos = self.view.mapToScene(event.position().toPoint())
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
                self.view.viewport().setCursor(Qt.ArrowCursor)
                return True

            elif event.button() == Qt.LeftButton:
                if "x:" in self.coord_label.text():
                    coords_text = self.coord_label.text().split(',')
                    x = int(coords_text[0].split(':')[1].strip())
                    y = int(coords_text[1].split(':')[1].strip())
                    
                    self._pixel_selected = (x, y)
                    self._render_image(preserve_zoom=True)

                    info = {
                        'x': x, 
                        'y': y, 
                        'obj_id': None, 
                        'obj_class': None, 
                        'value': None, 
                        'properties': {}
                    }
                    
                    if 0 <= y < self._original_image.shape[0] and 0 <= x < self._original_image.shape[1]:
                        pixel_val = self._original_image[y, x]
                        info['value'] = float(np.mean(pixel_val))
                    
                    obj_id = None
                    if 0 <= y < self._current_id_map.shape[0] and 0 <= x < self._current_id_map.shape[1]:
                        raw_id = self._current_id_map[y, x]
                        if raw_id >= 0 and not np.isnan(raw_id):
                            obj_id = int(raw_id)
                        info['obj_id'] = raw_id
                    
                    if 0 <= y < self._current_class_map.shape[0] and 0 <= x < self._current_class_map.shape[1]:
                        info['obj_class'] = int(self._current_class_map[y, x])

                    if obj_id in self._object_parameters:
                        info['properties'] = self._object_parameters[obj_id]

                    self.pixel_selected_data.emit(info)
                    return True
            
            elif event.button() == Qt.RightButton:
                self._pixel_selected = None
                self._render_image(preserve_zoom=True)
                self.pixel_selected_data.emit({})
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





    def _build_selection_overlay_array(self):
        if self._pixel_selected is None or self._current_id_map is None:
            return None
        
        h, w = self._current_id_map.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        x, y = self._pixel_selected
        if not (0 <= x < w and 0 <= y < h):
            return None
        
        overlay[y, x] = [255, 0, 0, 255]

        raw_id = self._current_id_map[y, x]
        if raw_id is None or np.isnan(raw_id) or raw_id < 0:
            return overlay
        
        obj_id = int(raw_id)
        ys, xs = np.nonzero(self._current_id_map == obj_id)
        overlay[ys, xs, :3] = [219, 255, 0]
        overlay[ys, xs, 3] = 200

        # Paint direct parent
        try:
            nodes = self._current_mto_struct.mt.contents.nodes
            img_data = self._current_mto_struct.mt.contents.img.data
            id_map_flat = self._current_id_map.ravel()
            p_val, p_obj, p_idx = self.get_direct_parent(obj_id, id_map_flat, nodes, img_data)

            if p_obj >= 0 and not np.isnan(p_obj):
                ys_p, xs_p = np.nonzero(self._current_id_map == p_obj)
                overlay[ys_p, xs_p, :3] = [0, 255, 255]
                overlay[ys_p, xs_p, 3] = 150
            
            if p_idx >= 0 and p_idx < id_map_flat.size:
                py, px = np.unravel_index(p_idx, self._current_id_map.shape)
                if 0 <= px < w and 0 <= py < h:
                    overlay[py, px] = [153, 0, 255, 255]
        except Exception:
            pass

        return overlay

import os
from astropy.io import fits
import numpy as np


from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem)
from PySide6.QtCore import Qt, Signal, Slot, QEvent
from PySide6.QtGui import QPainter, QPixmap, QImage, QCursor, QColor
from backend.classify import DIFFUSE, COMPACT


class ObjectInfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Object Information")
        self.setMinimumWidth(300)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.infoLabel = QLabel("Select an object to see its information.")
        self.infoLabel.setAlignment(Qt.AlignTop)
        self.layout.addWidget(self.infoLabel)


    @Slot(dict)
    def update_info(self, object_info: dict | None):
        if object_info is None:
            self.infoLabel.setText("No object selected.")
            return
        
        info_text = f"""
        """
        for key, value in object_info.get('properties', {}).items():
            info_text += f"<b>{key.capitalize()}:</b> {value}<br>"
        
        self.infoLabel.setText(info_text)

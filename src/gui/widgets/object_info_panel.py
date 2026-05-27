from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout)
from PySide6.QtCore import Qt, Slot, Signal
from backend.classify import DIFFUSE, COMPACT


class ObjectInfoPanel(QWidget):
    reclassify_requested = Signal(int, int)  # object_id, classification

    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.infoLabel = QLabel("Select an object to see its information.")
        self.infoLabel.setAlignment(Qt.AlignTop)
        self.layout.addWidget(self.infoLabel)

        self.current_object_id = None

        self.btn_layout = QHBoxLayout()
        self.btn_compact = QPushButton("Mark Compact")
        self.btn_diffuse = QPushButton("Mark Diffuse")
        self.btn_layout.addWidget(self.btn_compact)
        self.btn_layout.addWidget(self.btn_diffuse)
        self.layout.addLayout(self.btn_layout)

        self.btn_compact.clicked.connect(lambda: self.emit_classification(COMPACT))
        self.btn_diffuse.clicked.connect(lambda: self.emit_classification(DIFFUSE))
        
        self.btn_compact.setEnabled(False)
        self.btn_diffuse.setEnabled(False)


    @Slot(dict)
    def update_info(self, object_info: dict | None):
        if object_info is None:
            self.infoLabel.setText("No object selected.")
            self.current_object_id = None
            self.btn_compact.setEnabled(False)
            self.btn_diffuse.setEnabled(False)
            return

        object_id = object_info.get("obj_id", None)
        raw_classification = object_info.get("obj_class")
        if raw_classification is None:
            raw_classification = object_info.get("properties", {}).get("source_type")

        classification_labels = {
            DIFFUSE: "Diffuse",
            COMPACT: "Compact",
            -1: "Unclassified",
            None: "Unclassified",
        }
        classification_str = classification_labels.get(raw_classification, str(raw_classification))

        properties = dict(object_info.get("properties", {}))
        properties.pop("source_type", None)

        info_text = (
            f"<b>Object ID:</b> {object_id}<br>"
            f"<b>Source Type:</b> {classification_str}<br>"
            f"<b>X:</b> {object_info.get('x', '')}<br>"
            f"<b>Y:</b> {object_info.get('y', '')}<br>"
            f"<b>Value:</b> {object_info.get('value', '')}<br>"
        )

        for key, value in properties.items():
            info_text += f"<b>{key.replace('_', ' ').capitalize()}:</b> {value}<br>"

        
        self.infoLabel.setText(info_text)
        self.current_object_id = object_id
        self.btn_compact.setEnabled(True)
        self.btn_diffuse.setEnabled(True)


    def emit_classification(self, classification):
        if self.current_object_id is not None:
            self.reclassify_requested.emit(self.current_object_id, classification)
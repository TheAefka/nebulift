from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout)
from PySide6.QtCore import Qt, Slot, Signal
from backend.classify import DIFFUSE, COMPACT, NOISE


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

        self.layout.addStretch()
        self.layout.addWidget(QLabel("<b>Legend:</b>"))
        self.layout.addLayout(self._create_legend_item("red", "Compact"))
        self.layout.addLayout(self._create_legend_item("green", "Diffuse"))
        self.layout.addLayout(self._create_legend_item("blue", "Noise"))
        self.layout.addLayout(self._create_legend_item("gray", "Unclassified"))
        self.layout.addLayout(self._create_legend_item("yellow", "Selected Object"))
        self.layout.addLayout(self._create_legend_item("cyan", "Parent Object"))
        self.layout.addLayout(self._create_legend_item("purple", "Direct Parent"))


        self.btn_layout = QHBoxLayout()
        self.btn_compact = QPushButton("Mark (C)ompact")
        self.btn_diffuse = QPushButton("Mark (D)iffuse")
        self.btn_noise   = QPushButton("Mark (N)oise")
        self.btn_layout.addWidget(self.btn_compact)
        self.btn_layout.addWidget(self.btn_diffuse)
        self.btn_layout.addWidget(self.btn_noise)
        self.layout.addLayout(self.btn_layout)

        self.btn_compact.clicked.connect(lambda: self.emit_classification(COMPACT))
        self.btn_diffuse.clicked.connect(lambda: self.emit_classification(DIFFUSE))
        self.btn_noise.clicked.connect(lambda: self.emit_classification(NOISE))

        self.btn_compact.setShortcut("C")
        self.btn_diffuse.setShortcut("D")
        self.btn_noise.setShortcut("N")
        
        self.btn_compact.setEnabled(False)
        self.btn_diffuse.setEnabled(False)
        self.btn_noise.setEnabled(False)


    def _create_legend_item(self, color, label):
        row_layout = QHBoxLayout()
        color_rect = QLabel()
        color_rect.setFixedSize(20, 15)
        color_rect.setStyleSheet(f"background-color: {color}; border: 1px solid black;")
        row_layout.addWidget(color_rect)
        row_layout.addWidget(QLabel(label))
        return row_layout

    @Slot(dict)
    def update_info(self, object_info: dict | None):
        if object_info is None:
            self.infoLabel.setText("No object selected.")
            self.current_object_id = None
            self.btn_compact.setEnabled(False)
            self.btn_diffuse.setEnabled(False)
            self.btn_noise.setEnabled(False)
            return

        object_id = object_info.get("obj_id", None)
        raw_classification = object_info.get("obj_class")
        if raw_classification is None:
            raw_classification = object_info.get("properties", {}).get("source_type")

        classification_labels = {
            DIFFUSE: "Diffuse",
            COMPACT: "Compact",
            -1: "Unclassified",
            NOISE: "Noise",
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
            f"<b>Stretched Value:</b> {object_info.get('stretched_value', '')}<br>"
            "<br><b>Parent Object:</b><br>"
            f"<b>Parent ID:</b> {object_info.get('parent_id', '')}<br>"
            f"<b>Parent Classification:</b> {classification_labels.get(object_info.get('parent_class'), object_info.get('parent_class', ''))}<br>"
            "<br><b>Direct Parent:</b><br>"
            f"<b>Direct Parent X:</b> {object_info.get('direct_parent_x', '')}<br>"
            f"<b>Direct Parent Y:</b> {object_info.get('direct_parent_y', '')}<br>"
            f"<b>Direct Parent Value:</b> {object_info.get('direct_parent_value', '')}<br>"
            f"<b>Direct Parent Stretched Value:</b> {object_info.get('direct_parent_stretched_value', '')}<br>"
            "<br><b>Properties:</b><br>"
        )

        for key, value in properties.items():
            info_text += f"<b>{key.replace('_', ' ').capitalize()}:</b> {value}<br>"

        
        self.infoLabel.setText(info_text)
        self.current_object_id = object_id
        self.btn_compact.setEnabled(True)
        self.btn_diffuse.setEnabled(True)
        self.btn_noise.setEnabled(True)


    def emit_classification(self, classification):
        if self.current_object_id is not None:
            self.reclassify_requested.emit(self.current_object_id, classification)
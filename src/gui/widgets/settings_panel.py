import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QComboBox,
                               QSlider, QSpinBox)
from PySide6.QtCore import Qt



class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        self._setup_mtolib_group()
        self._setup_classification_group()
        self._setup_stretch_group()

        self.main_layout.addStretch()

    def _setup_mtolib_group(self):
        # MTOlib settings
        mtolibGroup = QGroupBox("MTOlib Settings")
        mtolibLayout = QFormLayout()

        self.mto_fits_path = QPushButton("Open")

        self.mto_alpha = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        self.mto_alpha.setRange(0.0, 1.0)
        self.mto_alpha.setValue(1e-6)
        
        self.mto_bg_mean_checkbox = QCheckBox()
        self.mto_bg_mean = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        self.mto_bg_mean.setRange(-1000000.0, 1000000.0)
        self.mto_bg_mean.setEnabled(False) # Disable input when checkbox unchecked 
        
        self.mto_bg_variance = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        self.mto_bg_variance.setRange(-1.0, 1000000.0)
        self.mto_bg_variance.setValue(-1.0)
        
        self.mto_gain = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        self.mto_gain.setRange(-1.0, 1000000.0)
        self.mto_gain.setValue(-1.0)

        self.mto_min_distance = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        
        self.mto_move_factor = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        self.mto_move_factor.setValue(0.5)
        
        self.mto_soft_bias = QDoubleSpinBox(decimals=6, value=0.0, singleStep=1e-6)

        self.mto_build_btn = QPushButton("(Re)build max-tree")
        self.mto_build_btn.setEnabled(False) # Disabled until fits path provided


        mtolibLayout.addRow("Input FITS", self.mto_fits_path)
        mtolibLayout.addRow("Alpha", self.mto_alpha)
        mtolibLayout.addRow("Custom background mean?", self.mto_bg_mean_checkbox)
        mtolibLayout.addRow("Background mean", self.mto_bg_mean)
        mtolibLayout.addRow("Background variance", self.mto_bg_variance)
        mtolibLayout.addRow("Gain", self.mto_gain)
        mtolibLayout.addRow("Minimum distance", self.mto_min_distance)
        mtolibLayout.addRow("Move factor", self.mto_move_factor)
        mtolibLayout.addRow("Soft bias", self.mto_soft_bias)
        mtolibLayout.addRow("Run MTObjects", self.mto_build_btn)
        
        mtolibGroup.setLayout(mtolibLayout)
        self.main_layout.addWidget(mtolibGroup)

        self.mto_bg_mean_checkbox.toggled.connect(self.mto_bg_mean.setEnabled)
        self.mto_fits_path.clicked.connect(self.open_image)


    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)",
        )
        if file_path:
            self.current_fits_path = file_path
            self.mto_build_btn.setEnabled(True)
            self.mto_fits_path.setText(os.path.basename(file_path))
            print(f"{file_path}")

    
    def _setup_classification_group(self):
        classifGroup = QGroupBox("Classification Settings")
        classifLayout = QFormLayout(classifGroup)

        self.classifControls = {}

        categories = ["R_whfm", "A/B"]

        for cat in categories:
            check, spin, row_ui = self._create_classification_row(cat)
            classifLayout.addRow(row_ui)

            self.classifControls[cat] = {"check": check, "spin": spin}        

        self.showClassificationOverlay = QCheckBox("Show Overlay?")
        classifLayout.addRow(self.showClassificationOverlay)

        self.main_layout.addWidget(classifGroup)


    def _create_classification_row(self, labelText):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)

        rowLayout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox(labelText)
        checkbox.setFixedWidth(80)
        spinbox = QDoubleSpinBox(decimals=4)
        spinbox.setRange(0.0, 100.0)

        spinbox.setEnabled(False)
        checkbox.toggled.connect(spinbox.setEnabled)

        rowLayout.addWidget(checkbox)
        rowLayout.addWidget(spinbox)

        return checkbox, spinbox, rowWidget


    def _setup_stretch_group(self):
        stretchGroup = QGroupBox("Stretch Settings")
        stretchLayout = QFormLayout(stretchGroup)

        self.stretchFuncComboBox = QComboBox()
        self.stretchFuncComboBox.addItems(["asinh"])
        stretchLayout.addRow("Stretch Function", self.stretchFuncComboBox)

        categories = ["Background", "Compact", "Diffuse", "Blackpoint"]

        for cat in categories:
            check, spin, row_ui = self._create_stretch_row(cat)
            stretchLayout.addRow(row_ui)

            self.classifControls[cat] = {"check": check, "spin": spin}        

        self.main_layout.addWidget(stretchGroup)



    def _create_stretch_row(self, labelText):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)

        rowLayout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(labelText)
        label.setFixedWidth(80)
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        spinbox = QSpinBox()
        slider.setRange(0, 2000)
        spinbox.setRange(0, 2000)

        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)

        rowLayout.addWidget(label)
        
        rowLayout.addWidget(slider)
        rowLayout.addWidget(spinbox)

        return slider, spinbox, rowWidget

import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QDoubleSpinBox,
                               QFileDialog, QHBoxLayout, QLabel, QComboBox,
                               QSlider, QSpinBox)
from PySide6.QtCore import Qt, Signal



class SettingsPanel(QWidget):

    open_requested = Signal()
    process_requested = Signal(str, dict, dict, dict)
    parameters_changed = Signal(dict, dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        self._setup_mtolib_group()
        self._setup_classification_group()
        self._setup_stretch_group()

        self.apply_btn = QPushButton("Apply Classification/Stretch")
        self.main_layout.addWidget(self.apply_btn)

        self.main_layout.addStretch()

        # self._connect_signals()
        self.apply_btn.clicked.connect(self._on_parameter_changed)

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
        self.mto_fits_path.clicked.connect(self.open_requested.emit)
        self.mto_build_btn.clicked.connect(self._emit_process_request)


    def update_fits_display(self, file_path):
        if file_path:
            self.current_fits_path = file_path
            self.mto_build_btn.setEnabled(True)
            self.mto_fits_path.setText(os.path.basename(file_path))

    
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

        categories = ["Background", "Compact", "Diffuse"]

        for cat in categories:
            check, spin, row_ui = self._create_stretch_row(cat)
            stretchLayout.addRow(row_ui)

            self.classifControls[cat] = {"check": check, "spin": spin}

        bp_slider, bp_spin, bp_row = self._create_stretch_row("Blackpoint", is_float=True, min_val=0.0, max_val=1.0, decimals=4)
        stretchLayout.addRow(bp_row)
        self.classifControls["Blackpoint"] = {"slider": bp_slider, "spin": bp_spin}

        self.main_layout.addWidget(stretchGroup)



    def _create_stretch_row(self, labelText, is_float=False, min_val=0, max_val=2000, decimals=4):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)
        rowLayout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(labelText)
        label.setFixedWidth(80)

        slider = QSlider(Qt.Horizontal)
        
        if is_float:
            spinbox = QDoubleSpinBox(decimals=decimals)
            spinbox.setRange(min_val, max_val)
            multiplier = 10**decimals
            spinbox.setSingleStep(1.0 / multiplier)
            slider.setRange(int(min_val * multiplier), int(max_val * multiplier))

            slider.valueChanged.connect(lambda v: spinbox.setValue(v / multiplier))
            spinbox.valueChanged.connect(lambda v: slider.setValue(int(v * multiplier)))
        else:
            spinbox = QSpinBox()
            spinbox.setRange(int(min_val), int(max_val))
            slider.setRange(int(min_val), int(max_val))

            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)

        rowLayout.addWidget(label)
        rowLayout.addWidget(slider)
        rowLayout.addWidget(spinbox)

        return slider, spinbox, rowWidget

    def _emit_process_request(self):
        if not hasattr(self, 'current_fits_path'):
            return
        
        # MTO settings
        mto_params = {
            'alpha': self.mto_alpha.value(),
            'move_factor': self.mto_move_factor.value(),
            'bg_mean': self.mto_bg_mean.value() if self.mto_bg_mean_checkbox.checkState() == Qt.Checked else None,
            'bg_variance': self.mto_bg_variance.value(),
            'gain': self.mto_gain.value(),
            'min_distance': self.mto_min_distance.value(),
            'move_factor': self.mto_move_factor.value(),
            'soft_bias': self.mto_soft_bias.value(),
        }

        # Classification settings
        classif_params = {
            'r_fwhm_threshold': self.classifControls["R_whfm"]["spin"].value()
        }

        # Stretch settings
        stretch_params = {
            'background': self.classifControls["Background"]["spin"].value(),
            'compact': self.classifControls["Compact"]["spin"].value(),
            'diffuse': self.classifControls["Diffuse"]["spin"].value(),
            'black_point': self.classifControls["Blackpoint"]["spin"].value(),
        }

        self.process_requested.emit(
            self.current_fits_path, 
            mto_params, 
            classif_params, 
            stretch_params
        )

    # def _connect_signals(self):
    #     for cat in self.classifControls:
    #         controls = self.classifControls[cat]
    #         if "spin" in controls:
    #             controls["spin"].valueChanged.connect(self._on_parameter_changed)
    #         if "slider" in controls:
    #             controls["slider"].valueChanged.connect(self._on_parameter_changed)

    def _on_parameter_changed(self):
        if not hasattr(self, 'current_fits_path'):
            return

        classif_params = {
            'r_fwhm_threshold': self.classifControls["R_whfm"]["spin"].value()
        }
        stretch_params = {
            'background': self.classifControls["Background"]["spin"].value(),
            'compact': self.classifControls["Compact"]["spin"].value(),
            'diffuse': self.classifControls["Diffuse"]["spin"].value(),
            'black_point': self.classifControls["Blackpoint"]["spin"].value(),
        }
        self.parameters_changed.emit(classif_params, stretch_params)
import os

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QCheckBox,
                               QFormLayout, QDoubleSpinBox, QHBoxLayout, QLabel,
                               QComboBox, QSlider, QSpinBox)
from PySide6.QtCore import Qt, Signal, QSignalBlocker

from gui.widgets.collapse import Collapse

THRESHOLD_CATEGORIES = ["R_fwhm", "A/B"]
STRETCH_CATEGORIES = [
    "Background",
    "Compact",
    "Diffuse",
    "Noise"
]


class SettingsPanel(QWidget):

    open_requested = Signal()
    process_requested = Signal(str, dict, dict, dict)
    stretch_requested = Signal(dict)
    classification_requested = Signal(dict, dict)
    parameters_changed = Signal(dict, dict)
    overlay_toggled = Signal(bool)
    save_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignTop)

        self._setup_mtolib_group()
        self._setup_classification_group()
        self._setup_stretch_group()

        self.main_layout.addStretch()

        self.save_btn = QPushButton("Save Image")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)

        self.main_layout.addWidget(self.save_btn)




    ###
    #
    # MTOlib settings panel
    #
    ###

    def _setup_mtolib_group(self):
        """
        Setup the MTOlib parameters in the settings panel. Default values are
        the ones used in pipeline.py and MTOlib example script.
        """
        mtolibGroup = Collapse("MTOlib Settings")
        mtolibLayout = QFormLayout()

        self.mto_fits_path = QPushButton("Open")
        mtolibLayout.addRow("Input FITS", self.mto_fits_path)

        self.mto_alpha = QDoubleSpinBox(decimals=6, singleStep=1e-6, value=1e-6,
                                        minimum=0.0, maximum=1.0)
        mtolibLayout.addRow("Alpha", self.mto_alpha)
        
        self.mto_bg_mean_checkbox = QCheckBox()
        self.mto_bg_mean = QDoubleSpinBox(decimals=6, singleStep=1e-6,
                                          minimum=-1000000.0, maximum=1000000.0)
        self.mto_bg_mean.setEnabled(False) # Disable input when unchecked 
        mtolibLayout.addRow("Custom background mean?", self.mto_bg_mean_checkbox)
        mtolibLayout.addRow("Background mean", self.mto_bg_mean)
        
        self.mto_bg_variance = QDoubleSpinBox(decimals=6, singleStep=1e-6,
                                              minimum=-1.0, maximum=1000000.0,
                                              value=-1.0)
        mtolibLayout.addRow("Background variance", self.mto_bg_variance)
        
        self.mto_gain = QDoubleSpinBox(decimals=6, singleStep=1e-6, minimum=-1.0,
                                       maximum=1000000.0, value=-1.0)
        mtolibLayout.addRow("Gain", self.mto_gain)

        self.mto_min_distance = QDoubleSpinBox(decimals=6, singleStep=1e-6)
        mtolibLayout.addRow("Minimum distance", self.mto_min_distance)
        
        self.mto_move_factor = QDoubleSpinBox(decimals=6, singleStep=1e-6, value=0.5)
        mtolibLayout.addRow("Move factor", self.mto_move_factor)

        self.mto_soft_bias = QDoubleSpinBox(decimals=6, value=0.0, singleStep=1e-6)
        mtolibLayout.addRow("Soft bias", self.mto_soft_bias)

        self.mto_build_btn = QPushButton("Build max-tree")
        self.mto_build_btn.setEnabled(False) # Disabled until fits path provided
        mtolibLayout.addRow("Run MTObjects", self.mto_build_btn)
        
        mtolibGroup.content_layout.addLayout(mtolibLayout)
        self.main_layout.addWidget(mtolibGroup)

        # Connect signals
        self.mto_bg_mean_checkbox.toggled.connect(self.mto_bg_mean.setEnabled)
        self.mto_fits_path.clicked.connect(self.open_requested.emit)
        self.mto_build_btn.clicked.connect(self._emit_process_request)


    def update_fits_display(self, file_path):
        """
        When a file path is provided, update the button text and enable the
        build button.
        """
        if file_path:
            self.current_fits_path = file_path
            self.mto_build_btn.setEnabled(True)
            self.mto_fits_path.setText(os.path.basename(file_path))

    


    ###
    #
    # Classification settings panel
    #
    ###


    def _setup_classification_group(self):
        """
        Setup the classification settings in the settings panel.
        """
        classifGroup = Collapse("Classification Settings")
        classifLayout = QFormLayout()

        self.classifControls = {}

        # Classifier selector dropdown
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems([
            "Thresholds",
            "GMM (log(R_fwhm), log(R_e), roundness)",
            "GMM (profile, concentration)",
            "LVQ",
        ])
        self.classifier_combo.currentIndexChanged.connect(self._on_classifier_changed)
        classifLayout.addRow("Classifier", self.classifier_combo)

        # Threshold controls
        for cat in THRESHOLD_CATEGORIES:
            check, spin, row_ui = self._create_classification_row(cat)
            classifLayout.addRow(row_ui)
            self.classifControls[cat] = {"check": check, "spin": spin, "row": row_ui}

        # LVQ hint label (only shown for LVQ)
        self.lvq_hint_label = QLabel("Label at least one object of each class to train the LVQ classifier.")
        self.lvq_hint_label.setWordWrap(True)
        self.lvq_hint_label.setVisible(False)
        classifLayout.addRow(self.lvq_hint_label)

        self.showClassificationOverlay = QCheckBox("Show Overlay?")
        self.showClassificationOverlay.toggled.connect(self.overlay_toggled)
        classifLayout.addRow(self.showClassificationOverlay)

        self.apply_classification_btn = QPushButton("Apply Classification")
        self.apply_classification_btn.setEnabled(False)
        self.apply_classification_btn.clicked.connect(self._emit_classification_request)
        classifLayout.addRow(self.apply_classification_btn)

        classifGroup.content_layout.addLayout(classifLayout)
        self.main_layout.addWidget(classifGroup)

    def _on_classifier_changed(self, index):
        is_threshold = index == 0
        is_lvq = index == 3
        for cat in THRESHOLD_CATEGORIES:
            self.classifControls[cat]["row"].setVisible(is_threshold)
        self.lvq_hint_label.setVisible(is_lvq)

    def selected_classifier(self) -> str:
        """Return a stable key for the selected classifier."""
        return ["threshold", "gmm1", "gmm2", "lvq"][self.classifier_combo.currentIndex()]


    def _create_classification_row(self, labelText):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)

        rowLayout.setContentsMargins(0, 0, 0, 0)

        checkbox = QCheckBox(labelText)
        checkbox.setFixedWidth(80)
        
        spinbox = QDoubleSpinBox(decimals=4, minimum=0.0, maximum=100.0)
        spinbox.setEnabled(False)
        
        checkbox.toggled.connect(spinbox.setEnabled)

        rowLayout.addWidget(checkbox)
        rowLayout.addWidget(spinbox)

        return checkbox, spinbox, rowWidget




    ###
    #
    # Stretch settings panel
    #
    ###

    def _setup_stretch_group(self):
        stretchGroup = Collapse("Stretch Settings")
        stretchLayout = QFormLayout()


        self.stretchControls = {}

        for name in STRETCH_CATEGORIES:
            sectionGroup = Collapse(name, toggled=False)
            sectionLayout = QFormLayout()

            # Stretch function dropdown
            stretchFuncComboBox = QComboBox()
            stretchFuncComboBox.addItems(["asinh", "linear"])
            sectionLayout.addRow("Stretch Function", stretchFuncComboBox)

            # Stretch factor slider
            stretch_slider, stretch_spin, stretch_row = self._create_stretch_row("Stretch Factor", is_float=True, min_val=0.0, max_val=1000.0, decimals=2)
            sectionLayout.addRow(stretch_row)

            # Black Point slider
            bp_slider, bp_spin, bp_row = self._create_stretch_row("Black Point", is_float=True, min_val=0.0, max_val=0.2, decimals=4)
            sectionLayout.addRow(bp_row)

            # Offset slider
            offset_slider, offset_spin, offset_row = self._create_stretch_row("Offset", is_float=True, min_val=-1.0, max_val=1.0, decimals=4)
            sectionLayout.addRow(offset_row)

            self.stretchControls[name] = {
                "function": stretchFuncComboBox,
                "stretch_factor": stretch_spin,
                "blackpoint": bp_spin,
                "offset": offset_spin
            }

            sectionGroup.content_layout.addLayout(sectionLayout)
            stretchLayout.addWidget(sectionGroup)

        self.apply_stretch_btn = QPushButton("Apply Stretch")
        self.apply_stretch_btn.setEnabled(False)
        self.apply_stretch_btn.clicked.connect(self._emit_stretch_request)
        stretchLayout.addRow(self.apply_stretch_btn)

        stretchGroup.content_layout.addLayout(stretchLayout)
        self.main_layout.addWidget(stretchGroup)



    def _create_stretch_row(self, labelText, is_float=False, min_val=0, max_val=1000, decimals=4):
        rowWidget = QWidget()
        rowLayout = QHBoxLayout(rowWidget)
        rowLayout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(labelText)
        label.setFixedWidth(85)

        slider = QSlider(Qt.Horizontal)
        
        if is_float:
            spinbox = QDoubleSpinBox(decimals=decimals)
            spinbox.setRange(min_val, max_val)
            spinbox.setKeyboardTracking(False)
            multiplier = 10**decimals
            spinbox.setSingleStep(1.0 / multiplier)
            slider.setRange(int(min_val * multiplier), int(max_val * multiplier))

            def sync_spinbox(value):
                with QSignalBlocker(spinbox):
                    spinbox.setValue(value / multiplier)

            def sync_slider(value):
                with QSignalBlocker(slider):
                    slider.setValue(int(round(value * multiplier)))

            slider.valueChanged.connect(sync_spinbox)
            spinbox.valueChanged.connect(sync_slider)
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




    ###
    #
    # Signals
    #
    ###


    def collect_stretch_parameters(self) -> dict:
        sp = {
            "stretch_factors": {},
            "blackpoints": {},
            "offsets": {},
            "functions": {}
        }
        for name in STRETCH_CATEGORIES:
            ctrl = self.stretchControls[name]
            sp["stretch_factors"][name] = ctrl["stretch_factor"].value()
            sp["blackpoints"][name] = ctrl["blackpoint"].value()
            sp["offsets"][name] = ctrl["offset"].value()
            sp["functions"][name] = ctrl["function"].currentText()
        return sp


    def collect_classification_parameters(self):
        thresholds = {}
        for cat in THRESHOLD_CATEGORIES:
            if self.classifControls[cat]["check"].isChecked():
                thresholds[cat] = self.classifControls[cat]["spin"].value()
            else:
                thresholds[cat] = None
        classif_params = {
            'classifier': self.selected_classifier(),
            'thresholds': thresholds
        }
        return classif_params



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
            'soft_bias': self.mto_soft_bias.value(),
        }

        self.process_requested.emit(
            self.current_fits_path, 
            mto_params, 
            self.collect_classification_parameters(), 
            self.collect_stretch_parameters()
        )
    
    def _emit_stretch_request(self):
        self.stretch_requested.emit(self.collect_stretch_parameters())


    def _emit_classification_request(self):
        self.classification_requested.emit(
            self.collect_classification_parameters(),
            self.collect_stretch_parameters()
        )
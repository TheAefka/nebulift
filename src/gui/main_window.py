import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                               QSlider, QProgressBar, QComboBox, QCheckBox,
                               QSplitter, QMessageBox, QDialog, QDialogButtonBox,
                               QLayout)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt

from gui.widgets.about_dialog import AboutDialog
from gui.widgets.settings_panel import SettingsPanel

# TODO: switch to Qt Resource Files (.qrc)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nebulift")
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))

        # Menu bar
        menuBar = self.menuBar()
        
        fileMenu = menuBar.addMenu("File")
        editMenu = menuBar.addMenu("Edit")
        helpMenu = menuBar.addMenu("Help")

        openAction = fileMenu.addAction("Open")
        exitAction = fileMenu.addAction("Exit")

        openAction.triggered.connect(lambda item: print(f"Action Open triggered"))
        exitAction.triggered.connect(self.close)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(self._aboutMessage)

        # Content
        container = QWidget()
        self.setCentralWidget(container)

        layout = QHBoxLayout(container)

        splitter = QSplitter()
        splitterLayout = QHBoxLayout(splitter)

        imageContainer = QWidget()
        imageLayout = QVBoxLayout(imageContainer)

        imageLabel = QLabel("Image Display WIP")
        imageLabel.setAlignment(Qt.AlignCenter)
        imageLayout.addWidget(imageLabel)

        leftPanel = SettingsPanel()
        
        layout.addWidget(splitter)
        splitterLayout.addWidget(leftPanel)
        splitterLayout.addWidget(imageContainer)
        splitter.setStretchFactor(1, 1)
        

    def _aboutMessage(self):
        dialog = AboutDialog(self)
        dialog.exec()

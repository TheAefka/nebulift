import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                               QSlider, QProgressBar, QComboBox, QCheckBox,
                               QSplitter, QMessageBox, QDialog, QDialogButtonBox,
                               QLayout, QFileDialog)
from PySide6.QtGui import QIcon, QPixmap, QKeySequence
from PySide6.QtCore import Qt

from gui.widgets.about_dialog import AboutDialog
from gui.widgets.settings_panel import SettingsPanel
from gui.widgets.image_panel import ImagePanel

# TODO: switch to Qt Resource Files (.qrc)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Nebulift")
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))
        self.resize(1280, 720)

        # Menu bar
        menuBar = self.menuBar()
        
        fileMenu = menuBar.addMenu("File")
        editMenu = menuBar.addMenu("Edit")
        viewMenu = menuBar.addMenu("View")
        helpMenu = menuBar.addMenu("Help")

        openAction = fileMenu.addAction("Open")
        exitAction = fileMenu.addAction("Exit")
        
        zoomInAction = viewMenu.addAction("Zoom In")
        zoomInAction.setShortcuts([QKeySequence("Ctrl+="), QKeySequence("Ctrl++")])
        zoomInAction.triggered.connect(self._zoom_in)
        zoomOutAction = viewMenu.addAction("Zoom Out")
        zoomOutAction.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoomOutAction.triggered.connect(self._zoom_out)

        openAction.triggered.connect(self.handle_open_image)
        exitAction.triggered.connect(self.close)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(self._aboutMessage)

        # Content
        container = QWidget()
        self.setCentralWidget(container)

        layout = QHBoxLayout(container)

        splitter = QSplitter()
        splitterLayout = QHBoxLayout(splitter)

        self.imagePanel = ImagePanel()
        self.leftPanel = SettingsPanel()

        self.leftPanel.open_requested.connect(self.handle_open_image)
        
        layout.addWidget(splitter)
        splitterLayout.addWidget(self.leftPanel)
        splitterLayout.addWidget(self.imagePanel)
        splitter.setStretchFactor(1, 1)
        

    def _aboutMessage(self):
        dialog = AboutDialog(self)
        dialog.exec()



    def handle_open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)",
        )
        if file_path:
            self.current_fits_path = file_path

            self.imagePanel.load_file(file_path)
            self.leftPanel.update_fits_display(file_path)

    def _zoom_in(self):
        self.imagePanel.zoom_in()

    def _zoom_out(self):
        self.imagePanel.zoom_out()
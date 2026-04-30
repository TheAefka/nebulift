import os

from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
                               QSlider, QProgressBar, QComboBox, QCheckBox,
                               QSplitter, QMessageBox, QDialog, QDialogButtonBox)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLayout

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

        leftPanel = QWidget()
        leftPanelLayout = QVBoxLayout(leftPanel)

        leftPanelLabel = QLabel("Settings Panel WIP")
        leftPanelLabel.setAlignment(Qt.AlignCenter)
        leftPanelLayout.addWidget(leftPanelLabel)

        splitterLayout.addWidget(leftPanel)
        layout.addWidget(splitter)
        splitterLayout.addWidget(imageContainer)
        

    def _aboutMessage(self):
        dialog = QDialog()
        dialog.setWindowTitle("Nebulift - About")
        dialog.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))
        iconLabel = QLabel()
        iconPixmap = QPixmap(os.path.join(BASE_DIR, "resources", "icons", "logo.png"))
        iconLabel.setPixmap(iconPixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        iconLabel.setAlignment(Qt.AlignVCenter)
        
        infoLabel = QLabel("""<h1>Nebulift</h1>
                              <span style='color: palette(dark); font-size: 11px;'>Version 0.0.1</span><br><br>
                              Nebulift is a tool for visualizing diffuse objects by<br>stretching FITS images without distorting point sources.<br><br>
                              <span style='color: palette(dark);'>Developed by Niels de Boer<br>
                              University of Groningen (2026)</span><br><br>
                              <a href="https://www.github.com/theaefka/nebulift">GitHub</a>""")
        infoLabel.setOpenExternalLinks(True)
        infoLabel.setAlignment(Qt.AlignVCenter)

        okButton = QDialogButtonBox(QDialogButtonBox.Ok)
        okButton.accepted.connect(dialog.accept)

        # Layout
        contentLayout = QHBoxLayout()
        contentLayout.addWidget(iconLabel)
        contentLayout.addWidget(infoLabel)

        mainLayout = QVBoxLayout(dialog)
        mainLayout.addLayout(contentLayout)
        mainLayout.addWidget(okButton)
        mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

        dialog.exec()
        


app = QApplication()
window = MainWindow()
window.show()
app.exec()
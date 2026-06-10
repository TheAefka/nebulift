import os
from PySide6.QtWidgets import (QDialog, QLabel, QVBoxLayout, QHBoxLayout,
                               QDialogButtonBox, QLayout)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtCore import Qt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Nebulift - About")
        self.setWindowIcon(QIcon(os.path.join(BASE_DIR, "resources", "icons", "logo.png")))
        iconLabel = QLabel()
        iconPixmap = QPixmap(os.path.join(BASE_DIR, "resources", "icons", "logo.png"))
        iconLabel.setPixmap(iconPixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        iconLabel.setAlignment(Qt.AlignVCenter)
        
        infoLabel = QLabel("""<h1>Nebulift</h1>
                              <span style='color: palette(dark); font-size: 11px;'>Version 0.1.8</span><br><br>
                              Nebulift is a tool for visualizing diffuse objects by<br>stretching FITS images without distorting point sources.<br><br>
                              <span style='color: palette(dark);'>Developed by Niels de Boer<br>
                              University of Groningen (2026)</span><br><br>
                              <a href="https://www.github.com/theaefka/nebulift">GitHub</a>""")
        infoLabel.setOpenExternalLinks(True)
        infoLabel.setAlignment(Qt.AlignVCenter)

        okButton = QDialogButtonBox(QDialogButtonBox.Ok)
        okButton.accepted.connect(self.accept)

        # Layout
        contentLayout = QHBoxLayout()
        contentLayout.addWidget(iconLabel)
        contentLayout.addWidget(infoLabel)

        mainLayout = QVBoxLayout(self)
        mainLayout.addLayout(contentLayout)
        mainLayout.addWidget(okButton)
        mainLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolButton, QGroupBox


class Collapse(QWidget):
    def __init__(self, title: str, parent=None, toggled=True):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        arrow = "▼" if toggled else "▶"
        self._toggle_btn = QToolButton(
            text=f"{arrow}  {title}",
            checkable=True,
            checked=toggled,
        )
        self._toggle_btn.setStyleSheet("QToolButton { border: none; }")
        self._toggle_btn.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle_btn)

        self._content_widget = QGroupBox()

        self._content_widget.setVisible(toggled)

        self.content_layout = QVBoxLayout(self._content_widget)
        self.content_layout.setContentsMargins(15, 5, 5, 5)

        layout.addWidget(self._content_widget)

    def _on_toggled(self, checked: bool):
        self._content_widget.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self._toggle_btn.setText(f"{arrow}  {self._toggle_btn.text()[3:]}")
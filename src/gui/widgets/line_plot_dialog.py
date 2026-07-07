import numpy as np
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QGroupBox
)
from PySide6.QtGui import QCloseEvent


class LinePlotDialog(QDialog):

    def __init__(
        self,
        original_image: np.ndarray,
        stretched_image: np.ndarray,
        x_start: int,
        x_end: int,
        y_start: int,
        y_end: int,
        x_index: int,
        y_index: int,
        class_map: np.ndarray = None,
        parent=None,
    ):
        super().__init__(parent)
        self.original_image = original_image
        self.stretched_image = stretched_image
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_index = x_index
        self.y_index = y_index
        self.class_map = class_map
        self._figure = None
        self._class_colors = {
            -1: np.array([45, 45, 45], dtype=np.uint8),
            0: np.array([0, 153, 0], dtype=np.uint8),
            1: np.array([200, 0, 0], dtype=np.uint8),
        }

        self.setWindowTitle("Line Plot")

        group = QGroupBox("Line selection")
        group_layout = QVBoxLayout(group)

        # Direction row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["Horizontal (row)", "Vertical (column)"])
        dir_row.addWidget(self.direction_combo)
        group_layout.addLayout(dir_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addWidget(buttons)

    def _line_profile(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2:
            if values.shape[-1] == 3:
                weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
                return values @ weights
            return values.mean(axis=-1)
        return values

    def _class_line(self, direction: int) -> np.ndarray:
        if self.class_map is None:
            return None

        if direction == 0:
            return self.class_map[self.y_index, self.x_start:self.x_end]
        return self.class_map[self.y_start:self.y_end, self.x_index]

    def _class_strip(self, class_line: np.ndarray) -> np.ndarray:
        if class_line is None or class_line.size == 0:
            return None

        strip = np.zeros((1, class_line.size, 3), dtype=np.uint8)
        for class_value, color in self._class_colors.items():
            mask = class_line == class_value
            if np.any(mask):
                strip[0, mask] = color
        return strip

    def _class_boundaries(self, class_line: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
        if class_line is None or class_line.size < 2:
            return np.array([], dtype=np.float64)

        boundary_indices = np.flatnonzero(class_line[1:] != class_line[:-1])
        return x_vals[boundary_indices] + 0.5


    def show_plot(self):

        direction = self.direction_combo.currentIndex()
        x_idx     = self.x_index
        y_idx     = self.y_index

        if direction == 0:
            original_line = self.original_image[y_idx, self.x_start:self.x_end]
            stretch_line  = self.stretched_image[y_idx, self.x_start:self.x_end]
            x_vals        = np.arange(self.x_start, self.x_end)
            x_label       = "Column (x)"
            title         = f"Horizontal line  y = {y_idx}"
        else:
            original_line = self.original_image[self.y_start:self.y_end, x_idx]
            stretch_line  = self.stretched_image[self.y_start:self.y_end, x_idx]
            x_vals        = np.arange(self.y_start, self.y_end)
            x_label       = "Row (y)"
            title         = f"Vertical line  x = {x_idx}"

        eps = 1e-10
        original_line = self._line_profile(original_line)
        original_plot = np.where(np.isfinite(original_line), original_line, eps)
        original_plot = np.clip(original_plot, eps, None)
        stretch_line = self._line_profile(stretch_line)
        stretch_plot = np.where(np.isfinite(stretch_line), stretch_line, eps)
        stretch_plot = np.clip(stretch_plot, eps, None)
        class_line = self._class_line(direction)
        class_strip = self._class_strip(class_line)

        fig = plt.figure(figsize=(11, 6))
        self._figure = fig
        fig.suptitle(title, fontsize=13)

        grid = fig.add_gridspec(2, 1, height_ratios=[4.0, 0.45], hspace=0.08)
        ax = fig.add_subplot(grid[0])
        ax_class = fig.add_subplot(grid[1], sharex=ax)

        original_axis = ax
        stretch_axis = ax.twinx()

        original_axis.step(x_vals, original_plot, where="mid", color="tab:blue", linewidth=0.9, label="Original")
        stretch_axis.step(x_vals, stretch_plot, where="mid", color="darkorange", linewidth=0.9, label="Stretched")
        original_axis.set_ylabel("Original intensity")
        stretch_axis.set_ylabel("Stretched intensity")
        original_axis.set_xlabel(x_label)
        original_axis.set_title("Line profile comparison")
        original_axis.grid(True, which="both", alpha=0.25)
        original_axis.tick_params(axis="y", colors="tab:blue")
        stretch_axis.tick_params(axis="y", colors="darkorange")

        legend_handles, legend_labels = original_axis.get_legend_handles_labels()
        stretch_handles, stretch_labels = stretch_axis.get_legend_handles_labels()
        original_axis.legend(
            legend_handles + stretch_handles,
            legend_labels + stretch_labels,
            loc="upper right",
        )

        if class_strip is not None:
            ax_class.imshow(
                class_strip,
                aspect="auto",
                interpolation="nearest",
                extent=[x_vals[0] - 0.5, x_vals[-1] + 0.5, 0, 1],
            )
            boundaries = self._class_boundaries(class_line, x_vals)
            for boundary in boundaries:
                original_axis.axvline(boundary, color="black", alpha=0.25, linewidth=0.8, linestyle="--")
                stretch_axis.axvline(boundary, color="black", alpha=0.25, linewidth=0.8, linestyle="--")
                ax_class.axvline(boundary, color="black", alpha=0.25, linewidth=0.8, linestyle="--")
            ax_class.set_yticks([])
            ax_class.set_xlabel("Classification: green diffuse, red compact, dark unclassified")
            ax_class.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax_class.spines["left"].set_visible(False)
            ax_class.spines["right"].set_visible(False)
            ax_class.spines["top"].set_visible(False)
        else:
            ax_class.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        try:
            from PySide6.QtWidgets import QApplication

            qt_app = QApplication.instance()
            if qt_app is not None:
                qt_app.aboutToQuit.connect(lambda: plt.close(fig))
        except Exception:
            pass

        plt.show(block=False)

    def closeEvent(self, event: QCloseEvent):
        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None
        super().closeEvent(event)
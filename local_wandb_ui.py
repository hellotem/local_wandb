#!/usr/bin/env python3
"""
Local-Wandb GUI – fully working tensor customization
"""

import sys, json, os, traceback, re
from pathlib import Path
from datetime import datetime
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QListWidget, QLabel, QMessageBox, QTextEdit, QGroupBox,
    QDialog, QSplitter, QAction, QMenuBar, QFileDialog, QPushButton,
    QSpinBox, QLineEdit, QFormLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPixmap, QPalette, QColor, QKeySequence

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt

from local_wandb import LocalWandb

# ------------------------------------------------------------------
def show_error(parent, title, msg, detail=None):
    dlg = QMessageBox(parent)
    dlg.setIcon(QMessageBox.Critical)
    dlg.setWindowTitle(str(title))
    dlg.setText(str(msg))
    if detail:
        dlg.setDetailedText(str(detail))
    dlg.exec_()

# ------------------------------------------------------------------
class TensorConfigDialog(QDialog):
    """Dialog to set bins & range for tensor heat-maps."""
    def __init__(self, parent=None, bins=40, vmin=None, vmax=None, num=None):
        super().__init__(parent)
        self.setWindowTitle("Customize Tensor")
        self.setFixedSize(260, 180)

        form = QFormLayout(self)

        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(5, 200)
        self.bins_spin.setValue(bins)
        form.addRow("Bins:", self.bins_spin)

        self.min_edit = QLineEdit(str(vmin) if vmin is not None else "")
        self.max_edit = QLineEdit(str(vmax) if vmax is not None else "")
        self.num_edit = QLineEdit(str(num) if num is not None else "")
        form.addRow("Min value (empty=auto):", self.min_edit)
        form.addRow("Max value (empty=auto):", self.max_edit)
        form.addRow("Downsample (empty=all):", self.num_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self):
        bins = self.bins_spin.value()
        vmin = float(self.min_edit.text()) if self.min_edit.text().strip() else None
        vmax = float(self.max_edit.text()) if self.max_edit.text().strip() else None
        num = int(self.num_edit.text()) if self.num_edit.text().strip() else None
        return bins, vmin, vmax, num

# ------------------------------------------------------------------
class LoggerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local-Wandb Viewer")
        self.resize(1500, 950)

        self.base_dir = "local_wandb"
        self.project = None
        self.lw_instances = {}
        self.settings = QSettings("local_logger", "ui")
        self.dark_mode = self.settings.value("dark", False, type=bool)

        # tensor customisation defaults
        self.tensor_bins = 40
        self.tensor_vmin = None
        self.tensor_vmax = None
        self.tensor_num = 10000
        self._last_tensor_name = None   # string, not QListWidgetItem

        self._build_ui()
        self._build_menu()
        self.populate_projects()
        self.apply_theme()

    # --------------------------------------------------------------
    # UI construction
    # --------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.splitter = QSplitter(Qt.Horizontal)

        # left panel
        self.left_widget = QWidget()
        self.left_widget.setMinimumWidth(250)
        self.left_widget.setMaximumWidth(260)
        lv = QVBoxLayout(self.left_widget)

        theme_lay = QHBoxLayout()
        theme_lay.addWidget(QLabel("Theme"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentIndex(1 if self.dark_mode else 0)
        self.theme_combo.currentIndexChanged.connect(self.toggle_theme)
        theme_lay.addWidget(self.theme_combo)
        lv.addLayout(theme_lay)

        project_lay = QHBoxLayout()
        project_lay.addWidget(QLabel("Project"))
        self.project_combo = QComboBox()
        self.project_combo.currentTextChanged.connect(self.load_runs)
        project_lay.addWidget(self.project_combo)
        lv.addLayout(project_lay)

        lv.addWidget(QLabel("Runs (multi-select)"))
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.run_list.itemSelectionChanged.connect(self.on_run_selection_changed)
        lv.addWidget(self.run_list, 1)

        for label, attr in (("Metrics", "metric_list"),
                            ("Images",  "image_list"),
                            ("Tensors", "tensor_list")):
            lv.addWidget(QLabel(label))
            lst = QListWidget()
            lst.setMaximumHeight(120)
            setattr(self, attr, lst)
            lv.addWidget(lst, 1)

        self.metric_list.itemClicked.connect(self.plot_metric)
        self.image_list.itemClicked.connect(self.show_image)
        self.tensor_list.itemClicked.connect(self.show_tensor_sequence)

        # action buttons
        btn_box = QGroupBox("Actions")
        bv = QVBoxLayout(btn_box)
        self.config_btn = QPushButton("Show Config")
        self.config_btn.clicked.connect(self.show_config)
        bv.addWidget(self.config_btn)

        self.terminal_btn = QPushButton("Show Terminal")
        self.terminal_btn.clicked.connect(self.show_terminal)
        bv.addWidget(self.terminal_btn)

        self.tensor_cfg_btn = QPushButton("Customize Tensor")
        self.tensor_cfg_btn.clicked.connect(self.open_tensor_config)
        bv.addWidget(self.tensor_cfg_btn)

        self.run_del_btn = QPushButton("Delete Runs")
        self.run_del_btn.clicked.connect(self.del_run)
        bv.addWidget(self.run_del_btn)
        lv.addWidget(btn_box)

        # right canvas
        self.figure = Figure(figsize=(6, 4))
        self.figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = Navi(self.canvas, self)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas, 1)

        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(right)
        self.setCentralWidget(self.splitter)

    # --------------------------------------------------------------
    # menu bar – File first, then View
    # --------------------------------------------------------------
    def _build_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_act = QAction("Open Folder…", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self.choose_base_dir)
        file_menu.addAction(open_act)

        reload_act = QAction("Reload Project", self)
        reload_act.setShortcut(QKeySequence("F5"))
        reload_act.triggered.connect(self.reload_project)
        file_menu.addAction(reload_act)

        rm_proj_act = QAction("Remove Project…", self)
        rm_proj_act.setShortcut(QKeySequence("Ctrl+Shift+Del"))
        rm_proj_act.triggered.connect(self.remove_project)
        file_menu.addAction(rm_proj_act)

        exit_act = QAction("Eixt", self)
        exit_act.setShortcut(QKeySequence("Ctrl+Q"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)


        view_menu = menubar.addMenu("View")
        toggle_act = QAction("Toggle Sidebar", self)
        toggle_act.setShortcut(QKeySequence("Alt+S"))
        toggle_act.triggered.connect(self.toggle_sidebar)
        view_menu.addAction(toggle_act)

    # --------------------------------------------------------------
    # base folder selection
    # --------------------------------------------------------------
    def choose_base_dir(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Select logger root folder", self.base_dir)
        if new_dir:
            self.base_dir = new_dir
            self.populate_projects()
            self.load_runs(None)

    # --------------------------------------------------------------
    # tensor customisation dialog + auto-redraw
    # --------------------------------------------------------------
    def open_tensor_config(self):
        dlg = TensorConfigDialog(self, self.tensor_bins, self.tensor_vmin, self.tensor_vmax, self.tensor_num)
        if dlg.exec_():
            self.tensor_bins, self.tensor_vmin, self.tensor_vmax, self.tensor_num = dlg.values()
            self._redraw_tensor(self._last_tensor_name)
    def del_run(self):
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs:
            QMessageBox.information(self, "Info", "No runs selected.")
            return

        reply = QMessageBox.question(
            self, "Confirm deletion",
            f"Delete the selected {len(runs)} run(s) data PERMANENTLY?\n{', '.join(runs)}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        for run in runs:
            path = Path(self.base_dir) / self.project / run
            if path.exists():
                try:
                    import shutil
                    shutil.rmtree(path)
                except Exception as e:
                    show_error(self, "Deletion failed", str(e))
        self.load_runs(self.project)

    def remove_project(self):
        if not self.project:
            QMessageBox.information(self, "Info", "No project selected.")
            return
        reply = QMessageBox.question(
            self, "Confirm deletion",
            f"Delete the entire project '{self.project}' and ALL its runs data PERMANENTLY?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        path = Path(self.base_dir) / self.project
        if path.exists():
            try:
                import shutil
                shutil.rmtree(path)
                QMessageBox.information(self, "Deleted", f"Project '{self.project}' removed.")
                self.project = None
                self.populate_projects()          # refresh drop-down
                if self.project_combo.count():
                    self.project_combo.setCurrentIndex(0)  # select first valid
                else:
                    self.load_runs(None)          # empty list
            except Exception as e:
                show_error(self, "Deletion failed", str(e))
        # --------------------------------------------------------------
    # theme / sidebar
    # --------------------------------------------------------------
    def toggle_theme(self, idx):
        self.dark_mode = idx == 1
        self.settings.setValue("dark", self.dark_mode)
        self.apply_theme()

    def apply_theme(self):
        app = QApplication.instance()
        if self.dark_mode:
            app.setStyle("Fusion")
            pal = QPalette()
            pal.setColor(QPalette.Window, QColor(30, 30, 30))
            pal.setColor(QPalette.WindowText, Qt.white)
            pal.setColor(QPalette.Base, QColor(42, 42, 42))
            pal.setColor(QPalette.Text, Qt.white)
            app.setPalette(pal)
        else:
            app.setStyle("Fusion")
            app.setPalette(app.style().standardPalette())

    def toggle_sidebar(self):
        self.left_widget.setVisible(not self.left_widget.isVisible())

    # --------------------------------------------------------------
    # project / run handling
    # --------------------------------------------------------------
    def populate_projects(self):
        self.project_combo.clear()
        root = Path(self.base_dir)
        if root.exists():
            self.project_combo.addItems(sorted([p.name for p in root.iterdir() if p.is_dir()]))

    def load_runs(self, project):
        self.project = project
        self.run_list.clear()
        for lst in (self.metric_list, self.image_list, self.tensor_list):
            lst.clear()
        self.lw_instances.clear()
        if not project:
            return
        try:
            runs = self._sorted_runs(project)
            self.run_list.addItems(runs)
        except Exception as e:
            show_error(self, "Load error", str(e))

    def reload_project(self):
        if self.project:
            self.load_runs(self.project)

    # --------------------------------------------------------------
    # run selection – keep lists mutually exclusive
    # --------------------------------------------------------------
    def _clear_other_lists(self, sender):
        for lst in (self.metric_list, self.image_list, self.tensor_list):
            if lst is not sender:
                lst.clearSelection()

    def on_run_selection_changed(self):
        if getattr(self, '_selection_in_progress', False):
            return
        self._selection_in_progress = True
        try:
            runs = [i.text() for i in self.run_list.selectedItems()]
            # clear lists
            self.metric_list.clear()
            self.image_list.clear()
            self.tensor_list.clear()
            if not runs:
                return

            metrics, images, tensors = set(), set(), set()

            QApplication.setOverrideCursor(Qt.WaitCursor)
            for r in runs:
                lw = self._get_lw(r)
                if not lw:
                    continue
                # metrics: only if dataframe exists
                if hasattr(lw, 'metrics_df'):
                    metrics.update(c for c in lw.metrics_df.columns if c != 'step')
                # images & tensors: always
                img_dir = lw.run_dir / "images"
                if img_dir.exists():
                    images.update(p.name for p in img_dir.glob("*.png"))
                for npy in lw.tensor_dir.glob("*.npy"):
                    name = npy.stem.partition("_step_")[0]
                    tensors.add(name)

                QApplication.processEvents()

            self.metric_list.addItems(sorted(metrics))
            self.image_list.addItems(sorted(images))
            self.tensor_list.addItems(sorted(tensors))
        finally:
            QApplication.restoreOverrideCursor()
            self._selection_in_progress = False

    # --------------------------------------------------------------
    # LocalWandb cache
    # --------------------------------------------------------------
    def _get_lw(self, run_name):
        if run_name not in self.lw_instances:
            try:
                self.lw_instances[run_name] = LocalWandb(
                    self.project, name=run_name,
                    base_dir=self.base_dir, mode="read"
                )
            except Exception as e:
                show_error(self, "Open error", str(e))
                return None
        return self.lw_instances[run_name]

    # --------------------------------------------------------------
    # plotting – mutually-exclusive list clicks
    # --------------------------------------------------------------
    def plot_metric(self, item):
        self._clear_other_lists(self.metric_list)
        metric = item.text()
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plotted = 0
        QApplication.setOverrideCursor(Qt.WaitCursor)
        for run in runs:
            lw = self._get_lw(run)
            if not lw or not hasattr(lw, 'metrics_df'):
                continue
            df = lw.metrics_df
            if metric not in df.columns:
                continue
            x = df['step'].to_numpy(dtype=float)
            y = df[metric].to_numpy(dtype=float)
            mask = ~np.isnan(y)
            if not np.any(mask):
                continue
            ax.plot(x[mask], y[mask], marker='o', label=run)
            plotted += 1
            QApplication.processEvents()
        QApplication.restoreOverrideCursor()
        if plotted == 0:
            return
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=.5)
        self.canvas.draw()

    def show_image(self, item):
        self._clear_other_lists(self.image_list)
        fname = item.text()
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs:
            return
        self.figure.clear()
        n = len(runs)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        for idx, run in enumerate(runs, start=1):
            lw = self._get_lw(run)
            if not lw:
                continue
            path = lw.run_dir / "images" / fname
            if not path.exists():
                continue
            ax = self.figure.add_subplot(rows, cols, idx)
            ax.imshow(plt.imread(path))
            ax.set_title(run, fontsize=8)
            ax.axis('off')
        self.canvas.draw()

    def show_tensor_sequence(self, item):
        self._clear_other_lists(self.tensor_list)
        tensor_name = item.text()
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs:
            return
        self._last_tensor_name = tensor_name
        self._redraw_tensor(tensor_name)

    # --------------------------------------------------------------
    # helper for tensor redrawing
    # --------------------------------------------------------------
    def _redraw_tensor(self, tensor_name):
        """Redraw tensor heat-map using current settings."""
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs or not tensor_name:
            return
        self.figure.clear()
        n = len(runs)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols
        QApplication.setOverrideCursor(Qt.WaitCursor)

        for idx, run in enumerate(runs, start=1):
            lw = self._get_lw(run)
            if not lw or tensor_name not in lw._tensor_buffers:
                continue
            steps, arrays = zip(*sorted(lw._tensor_buffers[tensor_name]))
            all_vals = np.concatenate(arrays)
            vmin = self.tensor_vmin if self.tensor_vmin is not None else float(all_vals.min())
            vmax = self.tensor_vmax if self.tensor_vmax is not None else float(all_vals.max())
            num = self.tensor_num if self.tensor_num is not None else 0
            if vmin == vmax:
                delta = abs(vmin) * 0.01 if vmin != 0 else 1.0
                vmin -= delta
                vmax += delta

            bins = self.tensor_bins
            bin_edges = np.linspace(vmin, vmax, bins + 1)
            hist_arr = []
            for arr in arrays:
                if num > 0 and num < arr.size:
                    arr_flat = arr.ravel()
                    arr_idx = np.random.choice(arr_flat.size, size=num, replace=False)
                    arr_sample = arr_flat[arr_idx]
                else:
                    arr_sample = arr
                hist_arr.append(np.histogram(arr_sample, bins=bin_edges)[0])
                QApplication.processEvents()
            heat = np.stack(hist_arr, axis=1)
            # heat = np.stack([np.histogram(arr, bins=bin_edges)[0] for arr in arrays], axis=1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            extent = [steps[0], steps[-1], bin_centers[0], bin_centers[-1]]

            ax = self.figure.add_subplot(rows, cols, idx)
            im = ax.imshow(heat, aspect='auto', origin='lower', cmap='OrRd',
                           extent=extent, interpolation='nearest')
            ax.set_title(run, fontsize=8)
            ax.set_xlabel("step")
            ax.set_ylabel("value")
            self.figure.colorbar(im, ax=ax, label='count')
        QApplication.restoreOverrideCursor()
        self.canvas.draw()

    # --------------------------------------------------------------
    # dialogs
    # --------------------------------------------------------------
    def _single_run(self):
        items = self.run_list.selectedItems()
        if len(items) != 1:
            QMessageBox.information(self, "Info", "Select exactly one run.")
            return None
        return self._get_lw(items[0].text())

    def show_config(self):
        lw = self._single_run()
        if not lw:
            return
        path = lw.run_dir / "config.json"
        if not path.exists():
            QMessageBox.information(self, "Config", "No config.json")
            return
        try:
            cfg = json.load(open(path))
            dlg = QDialog(self)
            dlg.setWindowTitle("Config")
            dlg.resize(600, 400)
            te = QTextEdit()
            te.setPlainText(json.dumps(cfg, indent=2))
            te.setReadOnly(True)
            v = QVBoxLayout(dlg)
            v.addWidget(te)
            dlg.exec_()
        except Exception as e:
            show_error(self, "Config error", str(e))

    def show_terminal(self):
        lw = self._single_run()
        if not lw:
            return
        path = lw.run_dir / "terminal.txt"
        if not path.exists():
            QMessageBox.information(self, "Terminal", "No terminal.txt")
            return
        try:
            text = ''.join(open(path).readlines()[-1000:])
            dlg = QDialog(self)
            dlg.setWindowTitle("Terminal output")
            dlg.resize(700, 500)
            te = QTextEdit()
            te.setPlainText(text)
            te.setReadOnly(True)
            v = QVBoxLayout(dlg)
            v.addWidget(te)
            dlg.exec_()
        except Exception as e:
            show_error(self, "Terminal error", str(e))

    # --------------------------------------------------------------
    # helper: sorted runs by datetime
    # --------------------------------------------------------------
    def _sorted_runs(self, project):
        proj_path = Path(self.base_dir) / project
        runs = []
        for d in proj_path.glob("run-*"):
            if d.is_dir():
                match = re.match(r"run-(\d{8}-\d{6})", d.name)
                if match:
                    runs.append((datetime.strptime(match.group(1), "%Y%m%d-%H%M%S"), d.name))
        runs.sort(key=lambda t: t[0])
        return [r[1] for r in runs]

# ------------------------------------------------------------------
if __name__ == "__main__":
    os.environ.pop("QT_QPA_PLATFORM", None)
    os.environ["QT_QPA_PLATFORM"] = "xcb"   # adjust to your platform
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    app = QApplication(sys.argv)
    matplotlib.use("Qt5Agg")   # now safe
    win = LoggerUI()
    win.show()
    sys.exit(app.exec_())
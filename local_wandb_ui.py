#!/usr/bin/env python3
"""
Local-Wandb GUI – mutually-exclusive list selection, multi-run plots
"""

import sys, json, os, traceback, re
from pathlib import Path
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QListWidget, QLabel, QMessageBox, QTextEdit, QGroupBox,
    QDialog, QSplitter, QAction, QMenuBar, QFileDialog, QPushButton
)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QPixmap, QPalette, QColor, QKeySequence

import matplotlib
matplotlib.use('Qt5Agg')
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
        self.left_widget.setMinimumWidth(300)
        self.left_widget.setMaximumWidth(360)
        lv = QVBoxLayout(self.left_widget)

        theme_lay = QHBoxLayout()
        theme_lay.addWidget(QLabel("Theme"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.setCurrentIndex(1 if self.dark_mode else 0)
        self.theme_combo.currentIndexChanged.connect(self.toggle_theme)
        theme_lay.addWidget(self.theme_combo)
        lv.addLayout(theme_lay)

        lv.addWidget(QLabel("Project"))
        self.project_combo = QComboBox()
        self.project_combo.currentTextChanged.connect(self.load_runs)
        lv.addWidget(self.project_combo)

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
        runs = [i.text() for i in self.run_list.selectedItems()]
        # clear lists
        self.metric_list.clear()
        self.image_list.clear()
        self.tensor_list.clear()
        if not runs:
            return

        metrics, images, tensors = set(), set(), set()
        for r in runs:
            lw = self._get_lw(r)
            if not lw:
                continue
            if hasattr(lw, 'metrics_df'):
                metrics.update(c for c in lw.metrics_df.columns if c != 'step')
            img_dir = lw.run_dir / "images"
            if img_dir.exists():
                images.update(p.name for p in img_dir.glob("*.png"))
            for npy in lw.tensor_dir.glob("*.npy"):
                name = npy.stem.partition("_step_")[0]
                tensors.add(name)

        self.metric_list.addItems(sorted(metrics))
        self.image_list.addItems(sorted(images))
        self.tensor_list.addItems(sorted(tensors))

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
        name = item.text()
        runs = [i.text() for i in self.run_list.selectedItems()]
        if not runs:
            return
        self.figure.clear()
        n = len(runs)
        cols = min(n, 2)
        rows = (n + cols - 1) // cols
        for idx, run in enumerate(runs, start=1):
            lw = self._get_lw(run)
            if not lw or name not in lw._tensor_buffers:
                continue
            steps, arrays = zip(*sorted(lw._tensor_buffers[name]))
            all_vals = np.concatenate(arrays)
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
            if vmin == vmax:
                delta = abs(vmin) * 0.01 if vmin != 0 else 1.0
                vmin -= delta
                vmax += delta

            bins = 40
            bin_edges = np.linspace(vmin, vmax, bins + 1)
            heat = np.stack([np.histogram(arr, bins=bin_edges)[0] for arr in arrays], axis=1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            extent = [steps[0], steps[-1], bin_centers[0], bin_centers[-1]]

            ax = self.figure.add_subplot(rows, cols, idx)
            im = ax.imshow(heat, aspect='auto', origin='lower', cmap='OrRd',
                           extent=extent, interpolation='nearest')
            ax.set_title(run, fontsize=8)
            ax.set_xlabel("step")
            ax.set_ylabel("value")
            self.figure.colorbar(im, ax=ax, label='count')
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
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
    app = QApplication(sys.argv)
    win = LoggerUI()
    win.show()
    sys.exit(app.exec_())
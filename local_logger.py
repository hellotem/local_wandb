import os
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
        self.lock = threading.Lock()

    def write(self, msg):
        with self.lock:
            for s in self.streams:
                s.write(msg)
                s.flush()

    def flush(self):
        with self.lock:
            for s in self.streams:
                s.flush()

class LocalWandb:
    def __init__(self, project: str, name: str = None, base_dir: str = "local_wandb", mode: str = "write"):
        """
        Initialize LocalWandb.
        mode="write": creates new run; mode="read": opens existing run with given name.
        """
        self.base_dir = Path(base_dir)
        self.project = project
        if mode == "write":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"run-{timestamp}" + (f"-{name}" if name else "")
            self.run_dir = self.base_dir / project / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._init_dirs()
            # Save run info and config
            with open(self.run_dir / "run_info.json", "w") as f:
                json.dump({"project": project, "name": name or "", "timestamp": timestamp}, f)
            self.config_path = self.run_dir / "config.json"
            # Metrics buffer
            self.metrics_records = []
            self.metrics_path = self.run_dir / "metrics.csv"
            self._step = 0
            self._tensor_buffers = {}
            self._mode = "write"

            # Set up terminal logging
            self.terminal_log_path = self.run_dir / "terminal.txt"
            self._terminal_log = open(self.terminal_log_path, "w")
            self._original_stdout = sys.stdout
            sys.stdout = TeeOutput(sys.stdout, self._terminal_log)

        elif mode == "read":
            self.run_dir = self.base_dir / project / name
            if not self.run_dir.exists():
                raise FileNotFoundError(f"Run {name} not found.")
            self._init_dirs()
            self.config_path = self.run_dir / "config.json"
            self.metrics_path = self.run_dir / "metrics.csv"
            self.metrics_df = pd.read_csv(self.metrics_path)
            # Load tensor buffers
            self._tensor_buffers = {}
            for npy in sorted(self.tensor_dir.glob("*.npy")):
                nm, _, step_str = npy.stem.partition("_step_")
                arr = np.load(npy)
                step = int(step_str)
                self._tensor_buffers.setdefault(nm, []).append((step, arr))
            for nm in self._tensor_buffers:
                self._tensor_buffers[nm].sort(key=lambda x: x[0])
            self._mode = "read"

        else:
            raise ValueError("mode must be 'write' or 'read'.")

    def _init_dirs(self):
        # Ensure directories exist
        self.images_dir = self.run_dir / "images"
        self.tensor_dir = self.run_dir / "tensors"
        for d in [self.images_dir, self.tensor_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def config(self, cfg: dict):
        """Save config dict (write mode only)."""
        if self._mode != "write":
            raise RuntimeError("Cannot write config in read mode.")
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def log(self, metrics: dict, step: int = None):
        """Log scalar metrics; supports dynamic keys."""
        if self._mode != "write":
            raise RuntimeError("Cannot log metrics in read mode.")
        if step is None:
            step = self._step
        # Record metrics
        record = {"step": step}
        record.update(metrics)
        self.metrics_records.append(record)
        # Write all records to CSV to update header dynamically
        pd.DataFrame(self.metrics_records).to_csv(self.metrics_path, index=False)
        self._step = step + 1

    def log_image(self, img: np.ndarray, name: str, cmap: str = None):
        """Save a numpy image array as PNG."""
        if self._mode != "write":
            raise RuntimeError("Cannot log images in read mode.")
        path = self.images_dir / f"{name}.png"
        plt.imsave(path, img, cmap=cmap)

    def log_figure(self, fig: plt.Figure, name: str, dpi: int = 150):
        """Save a Matplotlib figure as PNG."""
        if self._mode != "write":
            raise RuntimeError("Cannot log figures in read mode.")
        path = self.images_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    def log_tensor(self, name: str, tensor: torch.Tensor, step: int = None):
        """Log a tensor’s flattened values for histogram-over-time."""
        if self._mode != "write":
            raise RuntimeError("Cannot log tensors in read mode.")
        if step is None:
            step = self._step
        arr = tensor.detach().cpu().numpy().ravel()
        self._tensor_buffers.setdefault(name, []).append((step, arr))
        np.save(self.tensor_dir / f"{name}_step_{step}.npy", arr)

    def plot_metrics(self, keys=None, figsize=(8,4)):
        """Plot specified metrics over steps."""
        df = (pd.read_csv(self.metrics_path) if self._mode == "write"
              else self.metrics_df)
        keys = keys or [c for c in df.columns if c != 'step']
        for k in keys:
            if k in df.columns:
                plt.figure(figsize=figsize)
                plt.plot(df['step'], df[k], marker='o')
                plt.title(f"{k} over Steps")
                plt.xlabel('step')
                plt.ylabel(k)
                plt.grid(True)
                plt.show()

    def show_image(self, name: str, figsize=(6,6)):
        """Display a saved image by name."""
        fname = name if name.endswith('.png') else f"{name}.png"
        path = self.images_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Image '{fname}' not found.")
        img = Image.open(path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def plot_tensor_sequence(self, name: str, bins: int = 20, figsize=(6,4), cmap='OrRd'):
        """Show histogram-over-time for a tensor sequence."""
        if name not in self._tensor_buffers:
            print(f"No tensor '{name}' logged.")
            return
        data = self._tensor_buffers[name]
        steps, arrays = zip(*data)
        all_vals = np.concatenate(arrays)
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins+1)
        heat = np.stack([np.histogram(arr, bins=bin_edges)[0]
                         for arr in arrays], axis=1)
        plt.figure(figsize=figsize)
        plt.imshow(heat, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"Histogram over Time: {name}")
        plt.xlabel('step')
        plt.ylabel('value bins')
        plt.xticks(ticks=np.arange(len(steps)), labels=steps)
        plt.colorbar(label='count')
        plt.show()

    def show_config(self):
        """Print saved config JSON."""
        if self.config_path.exists():
            print(json.dumps(json.load(open(self.config_path)), indent=2))
        else:
            print("No config file found.")

    def show_terminal_output(self, lines: int = 20):
        """Print last N lines of captured terminal output."""
        if self.terminal_log_path.exists():
            with open(self.terminal_log_path) as f:
                print(''.join(f.readlines()[-lines:]))

    def finish(self):
        """Finalize logging: restore stdout and close files."""
        if self._mode == "write":
            # Restore stdout and close terminal log
            sys.stdout = self._original_stdout
            self._terminal_log.close()

    @staticmethod
    def list_runs(project: str, base_dir: str = "local_wandb"):
        """List all runs for a project in chronological order."""
        proj = Path(base_dir) / project
        if not proj.exists():
            return []
        runs = [d.name for d in proj.iterdir() if d.is_dir()]
        # Sort by timestamp in run-<timestamp>-name
        runs.sort(key=lambda name: name.split('-')[1])
        return runs

    @staticmethod
    def compare_metrics(project: str, metric: str, base_dir: str = "local_wandb", figsize=(8,4)):
        """Compare a specific metric across all runs in one plot."""
        runs = LocalWandb.list_runs(project, base_dir)
        plt.figure(figsize=figsize)
        for run in runs:
            path = Path(base_dir) / project / run / "metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if metric in df.columns:
                plt.plot(df['step'], df[metric], marker='o', label=run)
        plt.title(f"Comparison of '{metric}' across runs")
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

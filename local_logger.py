import os
import json
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

class LocalWandb:
    def __init__(self, project: str, name: str = None, base_dir: str = "local_wandb", mode: str = "write"):
        """
        Initialize logger.
        mode="write": create new run; mode="read": open existing run named `name`.
        """
        self.base_dir = Path(base_dir)
        if mode == "write":
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"run-{timestamp}" + (f"-{name}" if name else "")
            self.run_dir = self.base_dir / project / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._init_dirs()
            with open(self.run_dir / "run_info.json", "w") as f:
                json.dump({"project": project, "name": name or "", "timestamp": timestamp}, f)
            self.metrics_path = self.run_dir / "metrics.csv"
            self._metrics_file = open(self.metrics_path, "w", newline="")
            self._csv_writer = None
            self._step = 0
            self._tensor_buffers = {}
            self._mode = "write"
        elif mode == "read":
            self.run_dir = self.base_dir / project / name
            if not self.run_dir.exists():
                raise FileNotFoundError(f"Run {name} not found")
            self._init_dirs()
            self.config_path = self.run_dir / "config.json"
            self.metrics_path = self.run_dir / "metrics.csv"
            self.metrics_df = pd.read_csv(self.metrics_path)
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
            raise ValueError("mode must be 'write' or 'read'")

    def _init_dirs(self):
        self.images_dir = self.run_dir / "images"
        self.tensor_dir = self.run_dir / "tensors"
        for d in [self.images_dir, self.tensor_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.config_path = self.run_dir / "config.json"

    def config(self, cfg: dict):
        if self._mode != "write": raise RuntimeError("Read-only mode")
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def log(self, metrics: dict, step: int = None):
        if self._mode != "write": raise RuntimeError("Read-only mode")
        if step is None: step = self._step
        if self._csv_writer is None:
            header = ["step"] + list(metrics.keys())
            self._csv_writer = csv.DictWriter(self._metrics_file, fieldnames=header)
            self._csv_writer.writeheader()
        self._csv_writer.writerow({"step": step, **metrics})
        self._metrics_file.flush()
        self._step = step + 1

    def log_image(self, img: np.ndarray, name: str, cmap: str = None):
        if self._mode != "write": raise RuntimeError("Read-only mode")
        path = self.images_dir / f"{name}.png"
        plt.imsave(path, img, cmap=cmap)

    def log_figure(self, fig: plt.Figure, name: str, dpi: int = 150):
        if self._mode != "write": raise RuntimeError("Read-only mode")
        path = self.images_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)

    def log_tensor(self, name: str, tensor: torch.Tensor, step: int = None):
        if self._mode != "write": raise RuntimeError("Read-only mode")
        if step is None: step = self._step
        arr = tensor.detach().cpu().numpy().ravel()
        self._tensor_buffers.setdefault(name, []).append((step, arr))
        np.save(self.tensor_dir / f"{name}_step_{step}.npy", arr)

    def plot_metrics(self, keys=None, figsize=(8,4)):
        df = pd.read_csv(self.metrics_path) if self._mode=="write" else self.metrics_df
        keys = keys or [c for c in df.columns if c!='step']
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
        fname = name if name.endswith('.png') else f"{name}.png"
        path = self.images_dir / fname
        if not path.exists(): raise FileNotFoundError(fname)
        img = Image.open(path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title(name)
        plt.show()

    def plot_tensor_sequence(self, name: str, bins: int = 20, figsize=(6,4), cmap='OrRd'):
        if name not in self._tensor_buffers: return
        data = self._tensor_buffers[name]
        steps, arrays = zip(*data)
        all_vals = np.concatenate(arrays)
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins+1)
        heat = np.stack([np.histogram(arr, bins=bin_edges)[0] for arr in arrays], axis=1)
        plt.figure(figsize=figsize)
        plt.imshow(heat, aspect='auto', origin='lower', cmap=cmap)
        plt.title(f"Histogram over Time: {name}")
        plt.xlabel('step')
        plt.ylabel('value bins')
        plt.xticks(ticks=np.arange(len(steps)), labels=steps)
        plt.colorbar(label='count')
        plt.show()

    def show_config(self):
        if self.config_path.exists():
            print(json.dumps(json.load(open(self.config_path)), indent=2))

    def finish(self):
        if self._mode=="write": self._metrics_file.close()

    @staticmethod
    def list_runs(project: str, base_dir: str = "local_wandb"):
        proj = Path(base_dir) / project
        return [d.name for d in proj.iterdir() if d.is_dir()]


# === TEST SCRIPT ============================================================
def test_local_wandb():
    # Write mode demonstration
    run = LocalWandb("demo", "test", mode="write")
    run.config({"lr":0.01, "epochs":5, "batch_size":16})

    # Scalar logging
    for step in range(5):
        run.log({"loss": np.exp(-step/2), "acc": step/5}, step)

    # Image logging
    gradient = np.linspace(0,1,10000).reshape(100,100)
    run.log_image(gradient, "gradient", cmap='plasma')

    # Figure logging
    fig, ax = plt.subplots()
    x = np.linspace(0,10,100)
    ax.plot(x, np.sin(x), label='sin')
    ax.plot(x, np.cos(x), label='cos')
    ax.legend()
    ax.set_title('Trigonometric Functions')
    run.log_figure(fig, "trig_fig")

    # Tensor logging
    for step in range(5):
        t = torch.randn(500) * (1 + step*0.2)
        run.log_tensor("weights", t, step)

    run.finish()

    # Read mode demonstration
    runs = LocalWandb.list_runs("demo")
    reader = LocalWandb("demo", runs[-1], mode="read")
    print("Config:")
    reader.show_config()
    reader.plot_metrics()
    reader.show_image("gradient")
    reader.show_image("trig_fig")
    reader.plot_tensor_sequence("weights", bins=15)

if __name__ == "__main__":
    test_local_wandb()

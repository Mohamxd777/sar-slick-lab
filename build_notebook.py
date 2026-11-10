"""Package the source as a self-contained Kaggle notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULES = ("data_io.py", "models.py", "metrics.py", "run.py", "report.py")


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def main():
    cells = [
        cell("markdown", "# SAR Slick Lab\nFive-class oil-spill segmentation with U-Net and DeepLabV3+.\n"
             "Attach the `nabilsherif/oil-spill` dataset and select a GPU accelerator. Enable internet to fetch "
             "the torchvision ResNet-18 ImageNet weights. The cells below write the exact same modules as the local repo. "
             "The final run opens the official test split only after all four validation-selected models are trained."),
        cell("code", """import os
import sys
from pathlib import Path

import torch
assert torch.cuda.is_available(), "Select a GPU accelerator in Kaggle settings."
work = Path("/kaggle/working/sar-slick-lab")
work.mkdir(parents=True, exist_ok=True)
os.chdir(work)
dataset = Path("/kaggle/input/oil-spill")
if not dataset.is_dir():
    dataset = Path("/kaggle/input/datasets/nabilsherif/oil-spill")
assert dataset.is_dir(), "Attach the nabilsherif/oil-spill dataset before running."
print(torch.cuda.get_device_name(), dataset)"""),
    ]
    for name in MODULES:
        cells.append(cell("code", f"%%writefile {name}\n" + (ROOT / name).read_text(encoding="utf-8-sig")))
    cells.extend([
        cell("markdown", "## Train all four configurations and compare their held-out predictions"),
        cell("code", """import subprocess
subprocess.run([sys.executable, "run.py", "--data", str(dataset), "--out", "final",
                "--device", "cuda", "--epochs", "16", "--size", "256", "--batch", "4"], check=True)"""),
        cell("markdown", "## View results and save the report\nThe report and evidence files appear under the notebook output directory."),
        cell("code", """import importlib.util
import json
from IPython.display import Image, display

if importlib.util.find_spec("reportlab") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "reportlab>=4,<5"], check=True)
results = json.loads(Path("final/results.json").read_text())
print(json.dumps(results, indent=2))
display(Image(filename="final/dataset-preview.png"))
for row in results["experiments"]:
    print(row["experiment"])
    for filename in ["history.png", "confusion.png", "predictions.png"]:
        display(Image(filename=str(Path("final") / row["experiment"] / filename)))
subprocess.run([sys.executable, "report.py", "--run", "final",
                "--pdf", "SAR_Slick_Lab_Report.pdf"], check=True)"""),
    ])
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
                "language": "python", "name": "python3"}, "accelerator": "GPU"},
                "nbformat": 4, "nbformat_minor": 4}
    (ROOT / "SAR_Slick_Lab.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

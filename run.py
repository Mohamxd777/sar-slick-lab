"""Run the four controlled experiments and evaluate validation-selected weights."""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_io import COLORS, MEAN, NAMES, STD, Tiles, audit, split_pairs
from metrics import confusion, objective, scores
from models import DeepLabV3Plus, UNet

EXPERIMENTS = {"unet-ce": ("unet", False), "unet-balanced": ("unet", True),
               "deeplab-ce": ("deeplab", False), "deeplab-balanced": ("deeplab", True)}


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, data):
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def loader(pairs, args, training=False, epoch=0):
    return DataLoader(Tiles(pairs, args.size, training), batch_size=args.batch,
                      shuffle=training, num_workers=args.workers,
                      generator=torch.Generator().manual_seed(args.seed + epoch),
                      pin_memory=args.device == "cuda")


def visualize(panels, path):
    fig, axes = plt.subplots(len(panels), len(panels[0][1]), squeeze=False,
                             figsize=(4 * len(panels[0][1]), 3.2 * len(panels)))
    for row, (name, images) in zip(axes, panels):
        for ax, (title, image) in zip(row, images):
            ax.imshow(image, interpolation="nearest")
            ax.set_title(f"{name} | {title}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def image_for_display(image):
    return ((image.cpu() * STD + MEAN).clamp(0, 1)).permute(1, 2, 0).numpy()


def pass_epoch(model, batches, device, weights, optimizer=None, scaler=None, collect=False):
    model.train(optimizer is not None)
    total_loss, total_images = 0., 0
    matrix = torch.zeros((5, 5), dtype=torch.int64)
    selected = {}
    for x, y, names in batches:
        x, y = x.to(device), y.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            with torch.autocast(device, enabled=device == "cuda"):
                logits = model(x)
                loss = objective(logits, y, weights)
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        predictions = logits.detach().argmax(dim=1)
        matrix += confusion(predictions, y).cpu()
        total_loss += float(loss.detach()) * len(x)
        total_images += len(x)
        if collect:
            for image, target, pred, name in zip(x.cpu(), y.cpu().numpy(), predictions.cpu().numpy(), names):
                oil_errors = np.count_nonzero(((target == 1) & (pred == 2)) | ((target == 2) & (pred == 1)))
                ship_misses = np.count_nonzero((target == 3) & (pred != 3))
                for kind, count in [("First test tile", 1), ("Oil/look-alike errors", oil_errors), ("Missed ship pixels", ship_misses)]:
                    if count > selected.get(kind, (0, None))[0]:
                        selected[kind] = (count, (name, [("SAR", image_for_display(image)),
                                                        ("Ground truth", COLORS[target]), ("Prediction", COLORS[pred])]))
    return {"loss": total_loss / total_images, **scores(matrix)}, selected


def make_model(architecture, pretrained):
    return UNet() if architecture == "unet" else DeepLabV3Plus(pretrained)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, label in zip(axes, ("loss", "miou"), ("Objective loss", "Mean IoU")):
        for split in ("train", "validation"):
            ax.plot([row["epoch"] for row in history], [row[split][metric] for row in history],
                    marker="o", markersize=3, label=split)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(matrix, path):
    values = np.asarray(matrix)
    normalized = values / np.maximum(values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=NAMES, yticklabels=NAMES,
           xlabel="Predicted class", ylabel="True class")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for i, j in np.ndindex(5, 5):
        ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center",
                color="white" if normalized[i, j] > .55 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(args, name, train_pairs, validation_pairs, weights, split_manifest):
    architecture, balanced = EXPERIMENTS[name]
    weights = weights.to(args.device) if balanced else None
    directory = args.out / name
    directory.mkdir(exist_ok=True)
    latest = directory / "latest.pt"
    checkpoint = torch.load(latest, map_location="cpu", weights_only=True) if latest.exists() else None
    config = {"architecture": architecture, "balanced": balanced,
              "encoder_weights": "ImageNet-1K V1" if architecture == "deeplab" and not args.smoke else None,
              "size": args.size, "batch": args.batch, "seed": args.seed, "lr": args.lr,
              "device": args.device, "smoke": args.smoke, "split": split_manifest,
              "class_weights": weights.tolist() if balanced else None}
    if checkpoint and checkpoint["config"] != config:
        raise ValueError(f"Configuration changed for {name}. Use a new output directory.")
    seed(args.seed)
    model = make_model(architecture, config["encoder_weights"] is not None and checkpoint is None).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")
    history, best_score, best_epoch = [], -1., 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best_score, best_epoch = (checkpoint[key] for key in ("history", "best_score", "best_epoch"))
    if len(history) > args.epochs:
        raise ValueError("The output already contains more epochs than requested.")
    validation = loader(validation_pairs, args)
    for epoch in range(len(history) + 1, args.epochs + 1):
        start = time.perf_counter()
        seed(args.seed + epoch)
        tr, _ = pass_epoch(model, loader(train_pairs, args, True, epoch), args.device, weights, optimizer, scaler)
        val, _ = pass_epoch(model, validation, args.device, weights)
        history.append({"epoch": epoch, "train": tr, "validation": val, "seconds": time.perf_counter() - start})
        if val["miou"] > best_score:
            best_score, best_epoch = val["miou"], epoch
            save_checkpoint(directory / "best.pt", {"model": model.state_dict(), "config": config, "epoch": epoch})
        save_checkpoint(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                 "scaler": scaler.state_dict(), "history": history, "config": config,
                                 "best_score": best_score, "best_epoch": best_epoch})
        write_json(directory / "history.json", history)
        print(f"{name} epoch {epoch}: train mIoU={tr['miou']:.3f} val mIoU={val['miou']:.3f}", flush=True)
    write_json(directory / "history.json", history)
    write_json(directory / "config.json", {**config, "epochs": args.epochs})
    plot_history(history, directory / "history.png")
    return {"experiment": name, "best_epoch": best_epoch, "val_miou": best_score,
            "seconds": sum(row["seconds"] for row in history)}


def resized_weights(pairs, size):
    counts = np.zeros(5, dtype=np.int64)
    for _, mask, _ in Tiles(pairs, size):
        counts += np.bincount(mask.numpy().ravel(), minlength=5)
    if (counts == 0).any():
        raise ValueError(f"Every training class must survive resizing. Counts: {counts.tolist()}")
    weights = np.sqrt(counts.max() / counts).clip(1, 8)
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/final"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic fixture without encoder weights.")
    parser.add_argument("--train-only", action="store_true", help="Stop before opening the official test set.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, action="append",
                        help="Select training runs. Requires --train-only.")
    args = parser.parse_args()
    if args.experiment and not args.train_only:
        parser.error("--experiment requires --train-only.")
    if args.epochs < 1 or args.batch < 1 or args.size < 32 or args.size % 16:
        parser.error("Use positive epochs and batch size, and an image size divisible by 16 and at least 32.")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment.")
    if args.smoke != (args.data / "SYNTHETIC_FIXTURE").is_file():
        parser.error("Synthetic data must be explicitly marked and run with --smoke.")
    args.out.mkdir(parents=True, exist_ok=True)
    official = split_pairs(args.data, "train")
    if not args.smoke and len(official) != 1002:
        raise ValueError(f"Expected 1002 official training pairs, found {len(official)}.")
    permutation = np.random.default_rng(args.seed).permutation(len(official))
    n_val = max(1, round(.15 * len(official)))
    validation_pairs = [official[i] for i in permutation[:n_val]]
    train_pairs = [official[i] for i in permutation[n_val:]]
    split_manifest = {"train": [p[0].stem for p in train_pairs], "validation": [p[0].stem for p in validation_pairs]}
    weights, counts = resized_weights(train_pairs, args.size)
    write_json(args.out / "audit.json", {"train": audit(train_pairs), "validation": audit(validation_pairs),
                                       "resized_train_counts": counts, "class_names": NAMES, "rgb": COLORS.tolist()})
    samples = []
    for x, y, name in list(Tiles(validation_pairs[:4], args.size)):
        samples.append((name, [("SAR", image_for_display(x)), ("Ground truth", COLORS[y.numpy()])]))
    visualize(samples, args.out / "dataset-preview.png")
    rows = [train(args, name, train_pairs, validation_pairs, weights, split_manifest) for name in dict.fromkeys(args.experiment or EXPERIMENTS)]
    if args.train_only:
        return
    test = split_pairs(args.data, "test")
    if not args.smoke and len(test) != 110:
        raise ValueError(f"Expected 110 official test pairs, found {len(test)}.")
    write_json(args.out / "test-audit.json", audit(test))
    for row in rows:
        name = row["experiment"]
        directory = args.out / name
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model = make_model(checkpoint["config"]["architecture"], False).to(args.device)
        model.load_state_dict(checkpoint["model"])
        metrics, selected = pass_epoch(model, loader(test, args), args.device, None, collect=True)
        del metrics["loss"]
        row.update(metrics)
        plot_confusion(metrics["confusion"], directory / "confusion.png")
        visualize([panel for _, panel in selected.values()], directory / "predictions.png")
        row["examples"] = {kind: {"tile": panel[0], "pixels": int(count) if kind != "First test tile" else None}
                           for kind, (count, panel) in selected.items()}
    write_json(args.out / "results.json", {"synthetic": args.smoke, "test_images": len(test),
                                         "device": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                                         "experiments": rows})
    print(f"Results written to {args.out / 'results.json'}")


if __name__ == "__main__":
    main()

"""Pair official splits, audit exact labels, and prepare training tensors."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

NAMES = ("Sea", "Oil spill", "Look-alike", "Ship", "Land")
COLORS = np.array([(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)], dtype=np.uint8)
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def split_pairs(root, split):
    """Find exactly one split folder containing matching image and mask folders."""
    candidates = []
    for folder in [root, *root.rglob("*")]:
        if not folder.is_dir() or folder.name.lower() not in {split, split + "ing"}:
            continue
        subdirs = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
        images = next((subdirs[key] for key in ("images", "image", "imgs") if key in subdirs), None)
        masks = next((subdirs[key] for key in ("labels", "label", "masks", "mask", "annotations") if key in subdirs), None)
        if images is not None and masks is not None:
            candidates.append((images, masks))
    if len(candidates) != 1:
        raise ValueError(f"Expected one {split}/images + {split}/labels directory pair under {root}; found {candidates}")

    def indexed(folder):
        files = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in EXTENSIONS and path.is_file():
                key = path.stem.casefold()
                if key in files:
                    raise ValueError(f"Duplicate stem: {path}")
                files[key] = path
        return files

    images, masks = map(indexed, candidates[0])
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Unmatched {split} stems: {sorted(images.keys() ^ masks.keys())[:10]}")
    return [(images[key], masks[key]) for key in sorted(images)]


def decode(mask):
    """Accept real ID masks or decode palette/RGB masks by exact color."""
    raw = np.asarray(mask)
    if mask.mode != "P" and raw.ndim == 2 and raw.min() >= 0 and raw.max() < len(NAMES):
        return raw.astype(np.uint8)
    rgb = np.asarray(mask.convert("RGB"))
    ids = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for index, color in enumerate(COLORS):
        ids[np.all(rgb == color, axis=-1)] = index
    if np.any(ids == 255):
        unexpected = np.unique(rgb[ids == 255], axis=0)
        raise ValueError(f"Unknown mask colors: {unexpected[:10].tolist()}")
    return ids


def read_pair(pair):
    image_path, mask_path = pair
    with Image.open(image_path) as source, Image.open(mask_path) as mask:
        if source.size != mask.size:
            raise ValueError(f"Image and mask dimensions disagree: {image_path}, {mask_path}")
        try:
            return source.convert("RGB"), decode(mask)
        except ValueError as error:
            raise ValueError(f"{mask_path}: {error}") from error


def audit(pairs):
    counts = np.zeros(len(NAMES), dtype=np.int64)
    dimensions = {}
    for pair in pairs:
        image, ids = read_pair(pair)
        counts += np.bincount(ids.ravel(), minlength=len(NAMES))
        key = f"{image.width}x{image.height}"
        dimensions[key] = dimensions.get(key, 0) + 1
    return {"images": len(pairs), "dimensions": dimensions, "pixel_counts": counts.tolist()}


class Tiles(Dataset):
    def __init__(self, pairs, size, augment=False):
        self.pairs = pairs
        self.size = (size, size)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, ids = read_pair(self.pairs[index])
        image = np.asarray(image.resize(self.size, Image.Resampling.BILINEAR))
        ids = np.asarray(Image.fromarray(ids).resize(self.size, Image.Resampling.NEAREST))
        if self.augment:
            if torch.rand(()) < .5:
                image, ids = np.flip(image, axis=1), np.flip(ids, axis=1)
            turns = int(torch.randint(0, 4, ()).item())
            image, ids = np.rot90(image, turns), np.rot90(ids, turns)
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(ids.copy()).long()
        return (x - MEAN) / STD, y, self.pairs[index][0].stem

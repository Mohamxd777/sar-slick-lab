"""Create tiny synthetic RGB masks to verify the complete pipeline offline."""
from pathlib import Path

import numpy as np
from PIL import Image

from data_io import COLORS


def main(root=Path("data/synthetic")):
    rng = np.random.default_rng(4)
    for split, count in [("train", 10), ("test", 3)]:
        for folder in ("images", "labels"):
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            y = np.zeros((48, 48), dtype=np.uint8)
            offset = index % 5
            y[4:15, 3 + offset:16 + offset] = 1
            y[21:33, 13:26] = 2
            y[12 + offset:15 + offset, 32:35] = 3
            y[:, 39:] = 4
            intensity = np.array([110, 40, 57, 210, 170], dtype=float)[y]
            image = np.clip(intensity + rng.normal(0, 10, y.shape), 0, 255).astype(np.uint8)
            Image.fromarray(image).save(root / split / "images" / f"{split}-{index}.png")
            Image.fromarray(COLORS[y]).save(root / split / "labels" / f"{split}-{index}.png")
    (root / "SYNTHETIC_FIXTURE").touch()


if __name__ == "__main__":
    main()

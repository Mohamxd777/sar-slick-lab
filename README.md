# SAR Slick Lab

Five-class semantic segmentation of satellite synthetic aperture radar (SAR) imagery with a scratch U-Net and a DeepLabV3+ model that fine-tunes an ImageNet-pretrained encoder. The four controlled configurations measure the effect of changing the loss within each architecture.

**Verification status**  
All four paths ran end to end on a small generated CPU fixture. The official SAR dataset, ImageNet download, GPU fine-tuning, benchmark results and published Kaggle version are pending. The synthetic run uses a random DeepLab encoder and measures pipeline behavior only. See [the report](SAR_Slick_Lab_Report.pdf) and [Kaggle-ready notebook](SAR_Slick_Lab.ipynb).

## What the models learn

A SAR image represents radar return intensity. Oil can dampen small surface waves and produce a dark region, but similar dark returns can come from other phenomena. A ship can occupy only a few pixels. The target is one class ID per pixel, so the output can preserve where objects appear and show where oil is mistaken for a look-alike.

| ID | Class | RGB visualization |
|---:|---|---|
| 0 | Sea | (0, 0, 0) |
| 1 | Oil spill | (0, 255, 255) |
| 2 | Look-alike | (255, 0, 0) |
| 3 | Ship | (153, 76, 0) |
| 4 | Land | (0, 153, 0) |

The course brief lists these colors. The loader checks the actual pixel values when the official files are available. Unknown RGB colors fail with the mask filename and a list of unexpected values. Each output tensor has shape `[batch, 5, height, width]` and holds raw class scores (logits). Taking `argmax(dim=1)` yields a `[batch, height, width]` class mask. A color lookup translates class IDs back to RGB for inspection.

## Run locally

Use a Python environment with compatible PyTorch and torchvision builds. A GPU is recommended for the 16-epoch benchmark. Install a CUDA-enabled PyTorch build if the default installation on your system is CPU-only.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the [course dataset from Kaggle](https://www.kaggle.com/datasets/nabilsherif/oil-spill/data) and extract it to `data/oil-spill`. Additional wrapper folders are supported. Each official split needs image and mask directories whose files match by stem, such as `train/images/001.jpg` and `train/labels/001.png`. The official counts are 1,002 train pairs and 110 test pairs. Torchvision fetches the verified ResNet-18 weight file on the first DeepLab run unless it is already cached.

```powershell
python run.py --data data/oil-spill --out runs/final --epochs 16 --device cuda
python report.py --run runs/final --pdf SAR_Slick_Lab_Report.pdf
```

Add `--train-only` to postpone opening the official test split. Once the experiment settings and epoch budget are fixed, rerun the same command without that option to evaluate the four saved checkpoints. Reusing the output folder resumes completed epochs with matching configuration. Use a new folder for changed settings. Keep the official test set reserved for the final comparison.

### Offline check

```powershell
python smoke_fixture.py
python run.py --data data/synthetic --out runs/smoke --epochs 1 --batch 2 --size 32 --smoke
```

The fixture contains ten synthetic train pairs and three synthetic test pairs. A marker file separates them from the official path. Test scores from these generated tiles have no scientific interpretation for SAR.

### Kaggle notebook

Import [SAR_Slick_Lab.ipynb](SAR_Slick_Lab.ipynb), attach `nabilsherif/oil-spill`, select a GPU accelerator and enable Internet to fetch the ImageNet encoder weights. Use **Save Version → Save & Run All**. After the version finishes, open its logs and outputs. The notebook writes its work to `/kaggle/working/sar-slick-lab`, including checkpoints, history, plots, predictions and the report. Share the saved notebook version with the required audience using the dataset and course sharing rules. Rebuild the notebook from source with `python build_notebook.py` after editing the Python files.

## Four controlled experiments

| Run | Model | Loss |
|---|---|---|
| `unet-ce` | Scratch U-Net | Cross-entropy |
| `unet-balanced` | Scratch U-Net | Weighted cross-entropy + 0.5 × soft Dice loss |
| `deeplab-ce` | DeepLabV3+ with ImageNet-1K V1 ResNet-18 | Cross-entropy |
| `deeplab-balanced` | DeepLabV3+ with ImageNet-1K V1 ResNet-18 | Weighted cross-entropy + 0.5 × soft Dice loss |

All runs use the same seed (42), 85/15 tile-level split, 256 × 256 input size, training transforms, AdamW learning rate (0.001), weight decay (0.0001), batch size (4) and epoch budget (16). The DeepLab encoder is trainable from epoch one in both runs. The two runs within each architecture share their initialization procedure and data order. Compare each balanced run with its own CE baseline to isolate the **combined** change. The design cannot separate the effect of weighting from the effect of Dice. A cross-architecture difference also mixes architecture, capacity and pretraining.

### Loss and imbalance

Cross-entropy penalizes low probability assigned to the true class. Sea pixels dominate a pixel average when sea covers much of an image. For the balanced runs, the training set supplies per-class counts after resize. Class c receives weight `min(8, sqrt(max_count / count_c))`. The square root and cap temper the influence of extreme rarity. Soft Dice adds a differentiable overlap objective using predicted probabilities. It averages over the classes present in each batch, while CE still penalizes false positives for classes absent from the batch. The combined objective can improve minority recall or increase false alarms. Compare oil and ship IoU and the confusion matrices to distinguish those effects.

## Code walkthrough

### `data_io.py` — images and target masks

- `NAMES` and `COLORS` define the class order. `MEAN` and `STD` are ImageNet channel statistics with shape `[3,1,1]` so broadcasting applies them to each channel. `EXTENSIONS` filters supported raster files.
- `split_pairs` searches the dataset tree for a single official split. The nested `indexed` function matches filenames by stem, checks for duplicates, and catches missing partners before training starts. Sorting the stems makes the seeded split stable across directory listing order.
- `decode` accepts existing 0–4 ID rasters or resolves palette and RGB masks by exact color. Its temporary 255 value marks an unrecognized pixel. `read_pair` checks image/mask sizes and attaches the path to decode failures.
- `audit` counts original dimensions and original pixels per class. `Tiles.__getitem__` resizes imagery using bilinear interpolation and labels using nearest neighbor. Bilinear blends intensities, while nearest neighbor preserves discrete IDs. Flips and quarter-turns use the same random draw for image and mask. A copy ensures a contiguous NumPy buffer after transforms. The SAR image becomes a float CHW tensor normalized by ImageNet statistics. The target becomes an integer `long` tensor of shape `[H,W]` for cross-entropy.

### `models.py` — from image to logits

- `conv` composes convolution, group normalization and ReLU. Padding preserves height and width for stride-one layers. Group normalization works on channel groups and supports a single sample in the globally pooled ASPP branch. `double_conv` repeats the block. `upsample` interpolates features to an explicit target grid.
- `UNet` uses widths 24, 48, 96 and 192 across four encoder levels. Max-pooling halves the grid after each level. The 384-channel bottleneck processes the coarsest grid. Each decoder block upsamples, concatenates an encoder skip and refines the combined features. The skip recovers spatial detail that pooling can discard. A 1 × 1 head maps channels to five classes on the original grid.
- `separable` applies spatial filtering independently per channel, then a 1 × 1 convolution mixes channels. A dilation value spaces kernel samples apart to increase the field of view.
- `DeepLabV3Plus` keeps ResNet-18 feature layers and removes its image classification head. It replaces the final stride with dilation in the last stage so high-level features stay at stride 16. ASPP concatenates a 1 × 1 branch, three atrous branches and a global average-pooling branch. The context projection upsamples to stride four and joins a projected low-level feature map. The decoder refines those features and upsamples five logits to input size. Loading a trained checkpoint constructs an encoder without downloading ImageNet weights again because the checkpoint already contains them.

### `metrics.py` — objectives and performance

- `objective` takes raw logits and integer targets for CE. The balanced branch uses softmax, one-hot masks and per-class reductions over batch and spatial axes to calculate soft Dice.
- `confusion` encodes a true/predicted class pair as `5 × true + predicted` and bins all pixels into a 5 × 5 count matrix. Rows are ground truth and columns are predictions.
- `scores` derives true positives from the diagonal, true support from row totals and predicted support from column totals. IoU is TP / (TP + FP + FN), Dice is 2TP / (2TP + FP + FN), and accuracy is total correct pixels / total pixels. mIoU averages IoU over classes with nonzero union. A class absent from both truth and prediction has an undefined overlap and is stored as JSON null.

### `run.py` — reproducible orchestration

- `EXPERIMENTS` fixes the four configurations. `write_json` and `save_checkpoint` replace outputs only once their temporary files are written. `seed` initializes Python, NumPy and Torch randomness. `loader` batches samples and supplies an epoch-seeded generator for shuffling and worker seeds.
- `pass_epoch` toggles training mode and gradients based on whether an optimizer is supplied. The CUDA path uses mixed precision and a gradient scaler. It clears gradients, computes logits and loss, backpropagates for training, and accumulates pixel confusion across the entire split. During test evaluation it retains a first example and tiles with the most oil/look-alike errors and missed ship pixels when those failures occur.
- `image_for_display` undoes RGB normalization. `visualize` produces aligned image/mask rows. `plot_history` plots train/validation loss and mIoU. `plot_confusion` normalizes each ground-truth row so a rare class remains visible despite class imbalance.
- `train` initializes the model, optimizer and scaler, then resumes saved progress when possible. `best.pt` uses highest validation mIoU with earliest-epoch tie-breaking. `latest.pt` stores optimizer and scaler state for resumption. The config records the split and flags, and epoch JSON includes training and validation metrics and timings.
- `resized_weights` measures class frequencies at the grid the model actually sees and fails if resizing eliminates any training class. `main` audits the official training folder, creates train/validation, runs every configuration, then opens the official test folder. It evaluates the saved validation winners with the same preprocessing and emits the results bundle.

### Supporting files

- `smoke_fixture.py` creates repeatable noisy grayscale rectangles and aligned color masks so the pipeline can be exercised offline.
- `build_notebook.py` copies the exact source into notebook cells and adds the Kaggle setup, run command and display cells.
- `report.py` uses ReportLab to write methodology and status. Official-run JSON adds measured test and validation results, time, class scores, plots and selected failures. The smoke branch labels fixture results and omits SAR conclusions.
- `requirements.txt` lists runtime packages. `.gitignore` excludes local datasets, generated runs, model checkpoints and virtual environments.

## How to read the outputs

| Output | Read it for |
|---|---|
| `audit.json`, `test-audit.json` | Original image dimensions and pixel counts, plus resized train counts in the train audit |
| `dataset-preview.png` | Alignment and the effect of resizing on validation examples |
| `<run>/config.json` | Split, objective weights, encoder setup and training settings |
| `<run>/history.json`, `<run>/history.png` | Train/validation divergence, progress, loss behavior and per-epoch time |
| `<run>/best.pt`, `<run>/latest.pt` | Validation-selected weights and resumable state |
| `<run>/confusion.png`, `<run>/predictions.png` | Oil versus look-alike errors and missed ship pixels |
| `results.json` | mIoU, per-class IoU and Dice, pixel accuracy, confusion counts and representative tiles |

Look for a widening train/validation gap across several epochs before diagnosing overfitting. If both curves stagnate at low overlap, inspect whether small classes disappeared in preprocessing and whether the model learned any minority pixels. Compare loss magnitudes only within the same objective. The official metrics are computed on the resized grid. Small regions can disappear, and forcing all tiles to a square can distort shapes. A tile-level split may correlate near-duplicate scenes if group metadata is unavailable. One random seed gives a controlled comparison without estimating variability. GPU kernels and changes in hardware or versions can affect reproducibility.

## References and contributions

- [Krestenitis et al., Oil Spill Identification from Satellite Images Using Deep Neural Networks](https://doi.org/10.3390/rs11151762)
- [Ronneberger et al., U-Net](https://arxiv.org/abs/1505.04597)
- [Chen et al., DeepLabV3+](https://arxiv.org/abs/1802.02611)
- [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks)

Add verified names, responsibilities and evidence for all four or five team members before submission. This repository records the implementation and verification work without assigning unconfirmed contributions.

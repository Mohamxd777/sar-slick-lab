"""Generate the methodology report and add benchmark evidence when available."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as Figure, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TEAL = colors.HexColor("#06696c")
INK = colors.HexColor("#19303b")
EXPERIMENTS = [("U-Net CE", "Scratch U-Net", "Cross-entropy"),
               ("U-Net balanced", "Scratch U-Net", "Weighted CE + 0.5 Dice loss"),
               ("DeepLab CE", "ImageNet ResNet-18", "Cross-entropy"),
               ("DeepLab balanced", "ImageNet ResNet-18", "Weighted CE + 0.5 Dice loss")]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def format_score(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def build(run, destination):
    result_path = run / "results.json"
    results = load(result_path) if result_path.exists() else None
    synthetic = results is not None and results["synthetic"]
    benchmark = results is not None and not synthetic
    styles = getSampleStyleSheet()
    styles["Title"].fontSize, styles["Title"].leading = 30, 34
    styles["Title"].textColor = INK
    styles["Heading2"].textColor = TEAL
    styles["BodyText"].leading = 12
    story = []

    def paragraph(text, style="BodyText"):
        story.append(Paragraph(escape(text), styles[style]))

    def section(title, text):
        paragraph(title, "Heading2")
        paragraph(text)

    def table(rows, widths):
        grid = Table(rows, colWidths=[width * mm for width in widths], repeatRows=1, hAlign="LEFT")
        grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf5f5"), colors.white]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([Spacer(1, 5), grid, Spacer(1, 6)])

    def figure(path, max_height):
        with Image.open(path) as image:
            width, height = image.size
        scale = min(158 * mm / width, max_height * mm / height)
        story.append(Figure(str(path), width=width * scale, height=height * scale))

    paragraph("SAR Slick Lab", "Title")
    paragraph("AI II  /  Oil spill detection using semantic segmentation", "Heading2")
    if benchmark:
        paragraph(f"Status  /  Official held-out benchmark completed on {results['device']}.")
    elif synthetic:
        paragraph("Status  /  Pipeline verified on synthetic data. Official SAR benchmark pending.")
    else:
        paragraph("Status  /  Experimental methodology. Official SAR benchmark pending.")

    section("1. Objective and research background",
            "Segment sea, oil, look-alikes, ships and land in Synthetic Aperture Radar images. Oil slicks can dampen "
            "short surface waves, lowering radar backscatter. Other phenomena can also create dark regions. "
            "Krestenitis et al. (2019) introduced a labeled benchmark and found DeepLabV3+ strongest among the models "
            "they tested. The course dataset is expected to contain 1,002 training tiles and 110 official test tiles.")
    section("2. Data preparation and class imbalance",
            "Pair images and masks by filename stem and validate original dimensions and every RGB value before resizing. "
            "Accept integer labels 0 through 4 directly. Create a seeded 85/15 train-validation split from the "
            "official training folder. Apply bilinear image resize and nearest-neighbor mask resize. Use the same "
            "ImageNet RGB channel means and standard deviations for both models. Random horizontal flips and 90-degree "
            "rotations apply identically to training image and mask. Validation and test use no random transforms.")
    table([["ID", "Class", "RGB"], ["0", "Sea", "(0, 0, 0)"], ["1", "Oil spill", "(0, 255, 255)"],
           ["2", "Look-alike", "(255, 0, 0)"], ["3", "Ship", "(153, 76, 0)"], ["4", "Land", "(0, 153, 0)"]], [18, 55, 55])
    section("3. U-Net and DeepLabV3+",
            "U-Net starts from random weights. Four encoder blocks and max-pools precede a bottleneck. "
            "Bilinear decoder upsampling and skip concatenation restore fine spatial information. DeepLabV3+ "
            "starts its ResNet-18 encoder from torchvision ImageNet-1K V1 and fine-tunes all layers from epoch one. "
            "Atrous spatial pyramid pooling gathers multi-scale context, and a stride-four encoder skip restores "
            "local detail. Both heads return five logits per pixel on the input grid.")
    section("4. Controlled experiments",
            "Within each architecture, hold initialization, data split, augmentations, AdamW settings and epoch "
            "budget fixed. Compare plain cross-entropy with weighted cross-entropy plus soft Dice loss. Use capped "
            "inverse square-root frequency weights computed on resized training masks. Weighting increases minority "
            "class influence, and Dice directly rewards overlap. Rare-class weighting can also increase false alarms.")
    table([["Run", "Initialization", "Objective"], *EXPERIMENTS], [46, 51, 61])

    story.append(PageBreak())
    section("5. Monitoring and evaluation",
            "Select the earliest checkpoint with the highest validation mIoU. Record elapsed training time, loss, "
            "mIoU and class metrics each epoch. The official test set is loaded only after all four configurations "
            "are selected. IoU divides overlap by union, and Dice divides twice the overlap by the sum of both areas. "
            "Average IoU equally over classes with nonzero union. Pixel accuracy weights each pixel equally, so it "
            "can be high even when oil and ships are missed. Confusion rows represent ground truth.")
    if benchmark:
        audit = load(run / "audit.json")
        train_counts = audit["train"]["pixel_counts"]
        paragraph("Original training pixel proportions (Sea / Oil / Look-alike / Ship / Land)  " +
                  " / ".join(format_score(count / sum(train_counts)) for count in train_counts))
        paragraph("Training dimensions and class counts are available in audit.json.")
        table([["Experiment", "Epoch", "Val mIoU", "Test mIoU", "Accuracy", "Seconds"],
               *[[row["experiment"], row["best_epoch"], format_score(row["val_miou"]), format_score(row["miou"]),
                  format_score(row["accuracy"]), f"{row['seconds']:.1f}"] for row in results["experiments"]]],
              [44, 16, 25, 25, 25, 21])
        for metric in ("iou", "dice"):
            paragraph(f"Per-class test {metric.upper()}", "Heading3")
            table([["Experiment", "Sea", "Oil", "Look-alike", "Ship", "Land"],
                   *[[row["experiment"], *(format_score(x) for x in row[metric])] for row in results["experiments"]]],
                  [44, 21, 21, 27, 21, 21])
        ranked = max(results["experiments"], key=lambda row: row["miou"])
        paragraph(f"Highest held-out mIoU among the four fixed configurations was {ranked['experiment']} "
                  f"at {format_score(ranked['miou'])}. This is a descriptive test comparison.")
        for base, changed in zip(results["experiments"][::2], results["experiments"][1::2]):
            deltas = [changed["iou"][idx] - base["iou"][idx] if changed["iou"][idx] is not None
                      and base["iou"][idx] is not None else None for idx in (1, 3)]
            oil, ship = [f"{100 * x:+.2f} pp" if x is not None else "undefined" for x in deltas]
            paragraph(f"{changed['experiment']} vs. {base['experiment']}  |  oil IoU {oil}  |  ship IoU {ship}.")
    else:
        section("6. Results status",
                "Official mask audit, pretrained fine-tuning, held-out metrics, and SAR qualitative comparisons "
                "remain pending. The following verification uses generated geometric masks and noisy grayscale tiles."
                if synthetic else "Official dataset results and qualitative comparisons remain pending.")
        if synthetic:
            table([["Synthetic run", "Epoch", "Val mIoU", "Fixture mIoU", "Seconds"],
                   *[[row["experiment"], row["best_epoch"], format_score(row["val_miou"]),
                      format_score(row["miou"]), f"{row['seconds']:.1f}"] for row in results["experiments"]]],
                  [46, 20, 27, 32, 24])
            paragraph("The smoke test used one epoch per run, CPU, and random encoder weights for DeepLabV3+. "
                      "It exercises the implementation only. A one-point curve cannot establish convergence or overfitting.")

    section("7. Limitations and contribution statement",
            "Resizing can erase small ships and warp aspect ratios. A tile-level split may leave correlated scenes "
            "across training and validation when scene IDs are unavailable. A single seed does not estimate "
            "run-to-run variability. Hardware and library changes can affect reproducibility. "
            "Team names and individual contributions were not provided and require verification before submission.")
    section("References",
            "Krestenitis et al. (2019), Remote Sensing 11(15), 1762. DOI 10.3390/rs11151762. "
            "Kaggle course dataset: kaggle.com/datasets/nabilsherif/oil-spill. "
            "U-Net: arXiv:1505.04597. DeepLabV3+: arXiv:1802.02611.")
    if benchmark:
        story.append(PageBreak())
        paragraph("Dataset alignment preview", "Heading2")
        figure(run / "dataset-preview.png", 193)
        for row in results["experiments"]:
            story.append(PageBreak())
            name = row["experiment"]
            folder = run / name
            history = load(folder / "history.json")
            best = history[row["best_epoch"] - 1]
            last = history[-1]
            train_gain = last["train"]["miou"] - best["train"]["miou"]
            val_change = last["validation"]["miou"] - best["validation"]["miou"]
            paragraph(f"{name}  /  Learning curves", "Heading2")
            figure(folder / "history.png", 65)
            paragraph(f"From checkpoint epoch {row['best_epoch']} to final epoch {last['epoch']}, train mIoU changed "
                      f"{100 * train_gain:+.2f} pp and validation mIoU changed {100 * val_change:+.2f} pp.")
            if train_gain > .01 and val_change < -.01:
                paragraph("Training overlap improved after the selected epoch while validation declined, consistent "
                          "with a widening generalization gap. Inspect adjacent epochs for persistence.")
            elif last["epoch"] == row["best_epoch"]:
                paragraph("The highest validation mIoU was reached at the final epoch. Use the curves above to "
                          "inspect whether validation was still improving or had begun to flatten.")
            else:
                paragraph("The endpoint changes alone do not establish sustained divergence. Check the plotted "
                          "loss and mIoU trajectories for plateaus or oscillation.")
            figure(folder / "confusion.png", 105)
            matrix = row["confusion"]
            paragraph(f"Oil labeled as look-alike: {matrix[1][2]:,} pixels. Look-alike labeled as oil: "
                      f"{matrix[2][1]:,} pixels. Ships missed: {sum(matrix[3]) - matrix[3][3]:,} pixels.")
            story.append(PageBreak())
            paragraph(f"{name}  /  Held-out examples", "Heading2")
            figure(folder / "predictions.png", 177)
            for kind, example in row["examples"].items():
                paragraph(f"{kind}  |  {example['tile']}" +
                          (f"  |  {example['pixels']:,} pixels" if example["pixels"] is not None else ""))
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(destination), title="SAR Slick Lab", topMargin=14 * mm, bottomMargin=15 * mm)

    def footer(canvas, doc):
        canvas.setStrokeColor(TEAL)
        canvas.line(72, 38, doc.pagesize[0] - 72, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK)
        canvas.drawString(72, 24, "SAR SLICK LAB  /  AI II")
        canvas.drawRightString(doc.pagesize[0] - 72, 24, str(doc.page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/final"))
    parser.add_argument("--pdf", type=Path, default=Path("SAR_Slick_Lab_Report.pdf"))
    args = parser.parse_args()
    build(args.run, args.pdf)


"""Train a blood-cell detector from an annotated YOLO-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


CLASS_NAMES = ["red_blood_cell", "white_blood_cell", "platelet"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def validate_dataset(root: Path) -> Path:
    """Validate the expected images/{train,val} and labels/{train,val} layout."""
    root = root.expanduser().resolve()
    problems: list[str] = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            problems.append(f"Missing image folder: {image_dir}")
            continue
        images = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            problems.append(f"No supported images found in {image_dir}")
        if not label_dir.is_dir():
            problems.append(f"Missing label folder: {label_dir}")
            continue
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                problems.append(f"Missing YOLO label file: {label}")
                continue
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    problems.append(f"{label}:{line_no}: expected class_id x_center y_center width height")
                    continue
                try:
                    class_id = int(fields[0])
                    box = [float(value) for value in fields[1:]]
                except ValueError:
                    problems.append(f"{label}:{line_no}: label values must be numeric")
                    continue
                if class_id not in range(len(CLASS_NAMES)):
                    problems.append(f"{label}:{line_no}: class id must be 0, 1, or 2")
                if any(value < 0 or value > 1 for value in box):
                    problems.append(f"{label}:{line_no}: normalized box values must be in [0, 1]")
                if box[2] <= 0 or box[3] <= 0:
                    problems.append(f"{label}:{line_no}: box width and height must be positive")

    if problems:
        formatted = "\n".join(f"- {problem}" for problem in problems[:30])
        if len(problems) > 30:
            formatted += f"\n- ... and {len(problems) - 30} more issue(s)"
        raise ValueError("Dataset validation failed:\n" + formatted)

    data_file = root / "blood_cells.yaml"
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    if (root / "images" / "test").is_dir():
        config["test"] = "images/test"
    data_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return data_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO blood-cell detector")
    parser.add_argument("--data", type=Path, default=Path("dataset"), help="Dataset root with images/ and labels/ train/val folders")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO checkpoint or model yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Optional Ultralytics device, e.g. cpu, 0, or mps")
    parser.add_argument("--project", type=Path, default=Path.cwd() / "runs" / "detect")
    parser.add_argument("--name", default="blood_cell_counter")
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("epochs, imgsz, and batch must be positive (imgsz at least 32)")
    try:
        data_yaml = validate_dataset(args.data)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Using dataset config: {data_yaml}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    model = YOLO(args.model)
    train_options = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "plots": True,
    }
    if args.device:
        train_options["device"] = args.device
    model.train(**train_options)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best checkpoint: {best.resolve()}")
    print("The app will use this checkpoint automatically when it is present.")


if __name__ == "__main__":
    main()

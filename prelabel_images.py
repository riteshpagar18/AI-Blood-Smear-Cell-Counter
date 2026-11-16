"""Create draft YOLO labels and review overlays for microscope images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_NAMES = {"red_blood_cell", "white_blood_cell", "platelet"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("personal_data/images/train"))
    parser.add_argument("--labels", type=Path, default=Path("personal_data/labels/train"))
    parser.add_argument("--review", type=Path, default=Path("personal_data/review/train"))
    parser.add_argument("--model", type=Path, default=Path("runs/detect/txl_pbc_blood_cell_counter/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.20, help="Draft detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing label files")
    args = parser.parse_args()

    if not args.images.is_dir():
        parser.error(f"Image folder does not exist: {args.images}")
    if not args.model.is_file():
        parser.error(f"Model checkpoint does not exist: {args.model}")
    if not 0 < args.conf < 1 or args.imgsz < 32:
        parser.error("conf must be between 0 and 1 and imgsz must be at least 32")

    images = sorted(path for path in args.images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"No supported images found in {args.images}")
    duplicate_stems = {image.stem for image in images if sum(other.stem == image.stem for other in images) > 1}
    if duplicate_stems:
        parser.error(f"Images have duplicate filename stems; rename them first: {', '.join(sorted(duplicate_stems))}")

    model = YOLO(str(args.model))
    model_names = model.names
    names = list(model_names.values()) if isinstance(model_names, dict) else list(model_names)
    normalized = {str(name).strip().lower() for name in names}
    if normalized != EXPECTED_NAMES:
        parser.error(f"Checkpoint classes must be {sorted(EXPECTED_NAMES)}; found {names}")

    args.labels.mkdir(parents=True, exist_ok=True)
    args.review.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for image_path in images:
        label_path = args.labels / f"{image_path.stem}.txt"
        review_path = args.review / f"{image_path.stem}.jpg"
        if label_path.exists() and not args.overwrite:
            print(f"Skipping existing label: {label_path}")
            skipped += 1
            continue

        prediction = model.predict(
            source=str(image_path), conf=args.conf, imgsz=args.imgsz, verbose=False
        )[0]
        height, width = prediction.orig_shape
        rows: list[str] = []
        if prediction.boxes is not None:
            for box in prediction.boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                x_center = min(1.0, max(0.0, ((x1 + x2) / 2) / width))
                y_center = min(1.0, max(0.0, ((y1 + y2) / 2) / height))
                box_width = min(1.0, max(0.0, (x2 - x1) / width))
                box_height = min(1.0, max(0.0, (y2 - y1) / height))
                if box_width > 0 and box_height > 0:
                    rows.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        overlay = prediction.plot()
        cv2.imwrite(str(review_path), overlay)
        written += 1

    print(f"Drafted labels for {written} image(s); skipped {skipped} existing label(s).")
    print(f"Review overlays in: {args.review.resolve()}")
    print("Every draft must be reviewed and corrected before training.")


if __name__ == "__main__":
    main()

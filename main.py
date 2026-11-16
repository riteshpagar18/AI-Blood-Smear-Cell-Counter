"""Interactive blood-smear cell counter built with Streamlit and OpenCV.

This is a classical image-processing demo, not a clinically validated model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


@dataclass
class DetectionResult:
    annotated: np.ndarray
    counts: dict[str, int]
    boxes: dict[str, list[tuple[int, int, int, int]]]


COLORS = {
    "Red blood cells": (40, 210, 255),
    "White blood cells": (255, 100, 230),
    "Platelets": (70, 255, 110),
}
PROJECT_ROOT = Path(__file__).resolve().parent
DEPLOYED_MODEL = PROJECT_ROOT / "models" / "blood_cell_counter.pt"


def _remove_overlaps(circles: list[tuple[int, int, int]], distance: float) -> list[tuple[int, int, int]]:
    """Keep the larger of nearby Hough circle proposals."""
    kept: list[tuple[int, int, int]] = []
    for circle in sorted(circles, key=lambda c: c[2], reverse=True):
        if all(np.hypot(circle[0] - other[0], circle[1] - other[1]) >= distance for other in kept):
            kept.append(circle)
    return kept


def analyze_image(
    image: np.ndarray,
    min_cell_radius: int = 8,
    max_cell_radius: int = 34,
    sensitivity: int = 34,
    purple_threshold: int = 125,
) -> DetectionResult:
    """Detect approximate RBC circles and purple-stained WBC/platelet regions.

    ``image`` is an RGB uint8 array. All detections are heuristic and should be
    reviewed against the source image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel RGB image.")
    rgb = np.ascontiguousarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    found = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12, min_cell_radius * 1.35),
        param1=100,
        param2=sensitivity,
        minRadius=min_cell_radius,
        maxRadius=max_cell_radius,
    )
    circles = [] if found is None else [tuple(map(int, c)) for c in np.round(found[0]).astype(int)]
    circles = _remove_overlaps(circles, min_cell_radius * 1.2)
    height, width = gray.shape
    rbc_boxes: list[tuple[int, int, int, int]] = []
    for x, y, radius in circles:
        left, top = max(0, x - radius), max(0, y - radius)
        right, bottom = min(width, x + radius), min(height, y + radius)
        if right > left and bottom > top:
            rbc_boxes.append((left, top, right - left, bottom - top))

    # Purple/magenta stain has relatively high red and blue, with less green.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    purple = (((hue <= 175) & (hue >= 112)) | (hue <= 8)) & (sat >= 38) & (val >= 35)
    # The blue-vs-green channel difference helps with purple nuclei under varied hue.
    blue, green, red = cv2.split(bgr)
    purple |= (blue.astype(np.int16) - green.astype(np.int16) > 10) & (red.astype(np.int16) - green.astype(np.int16) > 5)
    purple_mask = (purple.astype(np.uint8) * 255)
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(purple_mask, 8)

    # Nuclei-sized purple components indicate likely WBCs. Smaller isolated
    # stain-positive particles are counted as platelet candidates.
    wbc_boxes: list[tuple[int, int, int, int]] = []
    platelet_boxes: list[tuple[int, int, int, int]] = []
    min_wbc_area = max(100, purple_threshold * 4)
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < 4:
            continue
        aspect = w / max(h, 1)
        if area >= min_wbc_area and area < height * width * 0.08 and 0.25 <= aspect <= 4:
            wbc_boxes.append((x, y, w, h))
        elif 4 <= area < min_wbc_area:
            platelet_boxes.append((x, y, w, h))

    # Draw cell classes in distinct colors. WBC outlines are shown above nuclei.
    canvas = bgr.copy()
    for x, y, w, h in rbc_boxes:
        cv2.ellipse(canvas, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, COLORS["Red blood cells"], 2)
    for x, y, w, h in platelet_boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["Platelets"], 1)
    for x, y, w, h in wbc_boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLORS["White blood cells"], 2)

    return DetectionResult(
        annotated=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
        counts={"Red blood cells": len(rbc_boxes), "White blood cells": len(wbc_boxes), "Platelets": len(platelet_boxes)},
        boxes={"Red blood cells": rbc_boxes, "White blood cells": wbc_boxes, "Platelets": platelet_boxes},
    )


@st.cache_resource
def load_detector(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def analyze_with_model(image: np.ndarray, weights_path: Path) -> DetectionResult:
    """Run the trained object detector and return per-class counts and overlay."""
    model = load_detector(str(weights_path.resolve()))
    prediction = model.predict(source=image, verbose=False)[0]
    names = model.names
    counts: dict[str, int] = {"Red blood cells": 0, "White blood cells": 0, "Platelets": 0}
    boxes_by_class: dict[str, list[tuple[int, int, int, int]]] = {name: [] for name in counts}
    display_names = {
        "red_blood_cell": "Red blood cells", "red blood cell": "Red blood cells", "rbc": "Red blood cells",
        "white_blood_cell": "White blood cells", "white blood cell": "White blood cells", "wbc": "White blood cells",
        "platelet": "Platelets", "platelets": "Platelets",
    }
    if prediction.boxes is not None:
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            raw_name = names[class_id] if isinstance(names, dict) else names[class_id]
            class_name = display_names.get(str(raw_name).strip().lower())
            if class_name is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            boxes_by_class[class_name].append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))
            counts[class_name] += 1
    annotated_bgr = prediction.plot()
    return DetectionResult(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), counts, boxes_by_class)


def main() -> None:
    st.set_page_config(page_title="AI Blood Smear Cell Counter", page_icon="🩸", layout="wide")
    st.title("🩸 AI Blood Smear Cell Counter")
    st.caption("Automated image analysis for red blood cells, white blood cells, and platelet candidates")
    st.info("This educational computer-vision demo is not a diagnostic tool. Review detections manually; image stain, focus, overlap, and magnification affect counts.")

    with st.sidebar:
        st.header("Detection settings")
        min_radius = st.slider("Minimum red cell radius (px)", 4, 25, 8)
        max_radius = st.slider("Maximum red cell radius (px)", 15, 80, 34)
        sensitivity = st.slider("Circle sensitivity", 15, 60, 34, help="Lower values detect more circles and may increase false positives.")
        purple_threshold = st.slider("Minimum WBC stain area", 50, 500, 125, 25, help="Raise this if stained debris is counted as a white blood cell.")
        st.caption("Tip: adjust settings after viewing the annotated image.")

    uploaded = st.file_uploader("Upload a microscope image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if uploaded is None:
        st.markdown("#### Get started")
        st.write("Upload a stained blood-smear image to detect approximate cell locations and see class-wise totals.")
        st.write("Supported formats: PNG, JPG, TIFF, and BMP.")
        return

    try:
        source = np.asarray(Image.open(uploaded).convert("RGB"))
        checkpoint_candidates = list((PROJECT_ROOT / "runs").glob("**/weights/best.pt"))
        if DEPLOYED_MODEL.is_file():
            checkpoint_candidates.append(DEPLOYED_MODEL)
        checkpoint = max(checkpoint_candidates, key=lambda candidate: candidate.stat().st_mtime) if checkpoint_candidates else None
        if checkpoint is not None:
            result = analyze_with_model(source, checkpoint)
            st.success(f"Using trained detector: `{checkpoint}`")
        else:
            result = analyze_image(source, min_radius, max(max_radius, min_radius + 1), sensitivity, purple_threshold)
            st.warning("No trained checkpoint found yet. Showing heuristic detections; train a model using the instructions in README.md.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Original image")
        st.image(source, use_container_width=True)
    with right:
        st.subheader("Detections")
        st.image(result.annotated, use_container_width=True)

    st.subheader("Cell counts")
    metric_cols = st.columns(3)
    for column, (name, value) in zip(metric_cols, result.counts.items()):
        column.metric(name, value)

    table = pd.DataFrame([{"Cell class": name, "Count": count} for name, count in result.counts.items()])
    st.dataframe(table, hide_index=True, use_container_width=True)
    stem = Path(uploaded.name).stem or "blood_smear"
    st.download_button("Download counts as CSV", table.to_csv(index=False).encode("utf-8"), f"{stem}_cell_counts.csv", "text/csv")
    st.caption("Outline colors: yellow = red blood cell, magenta = white blood cell, green = platelet candidate.")


if __name__ == "__main__":
    main()

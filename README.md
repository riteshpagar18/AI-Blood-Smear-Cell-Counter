# AI Blood Smear Cell Counter

A Python web app for blood-smear image analysis. It includes a YOLO object-detection training pipeline for multiple annotated images, model inference with class-wise totals, an annotated preview, and CSV export. Before a checkpoint is trained, the app shows its original OpenCV heuristic detections.

## Run in PyCharm

1. Open this folder as a PyCharm project and select its Python interpreter (the existing `.venv` is suitable).
2. In PyCharm's terminal, install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run main.py
   ```

4. Open the local URL printed in the terminal, upload a PNG/JPG/TIFF/BMP smear image, and adjust the sidebar settings if needed. Use **Download counts as CSV** to save the class totals.

## Train a model on your images

The project includes TXL-PBC with pre-existing annotations: 882 training, 252 validation, and 126 test images for RBCs, WBCs, and platelets. Dataset source, class mapping, and license details are in [`dataset/README.md`](dataset/README.md). The previous BCCD files are archived under `dataset/bccd_original/`; do not merge them because TXL-PBC includes selected BCCD images.

1. Keep the supplied train/validation/test splits, or add your own separately collected and annotated images using the structure shown in [`dataset/README.md`](dataset/README.md).
2. Install the training dependency (included in `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

3. Start training from the project root:

   ```bash
   python train_model.py --data dataset --model yolo26n.pt --epochs 25 --imgsz 512 --batch 24 --device mps --name txl_pbc_blood_cell_counter
   ```

   On Apple Silicon, `--device mps` may use the Metal GPU; use `--device cpu` to force CPU training. If your hardware runs out of memory, lower `--batch`.

4. Training writes metrics and the best checkpoint to `runs/detect/txl_pbc_blood_cell_counter/weights/best.pt`. The current run used 25 epochs at 512px with batch size 24 on MPS. Restart or refresh the app; it automatically uses the latest checkpoint for uploads.

### Current run results

On TXL-PBC's 252-image validation split, mAP@0.5 was 0.980. A one-time evaluation on the separate 126-image test split produced precision 0.956, recall 0.979, mAP@0.5 0.977, and mAP@0.5:0.95 0.852. Per-class mAP@0.5:0.95: RBC 0.895, WBC 0.888, platelets 0.774. The test set contains only 49 platelet boxes, so the platelet result is less certain. Inspect the prediction overlays and normalized confusion matrix before presenting the scores; the matrix shows background false positives concentrated in the RBC class. Evaluation plots and metrics are saved under `runs/detect/txl_pbc_test_eval/`.

Class IDs in annotation files must be `0` red blood cell, `1` white blood cell, and `2` platelet. Use an annotation tool that exports YOLO detection format. Start with representative labeled examples for every cell class, and keep images from the same slide or patient in only one split.

## Adapt to your own microscope

The TXL-PBC checkpoint is a starting point. To specialize it, collect images from your microscope using the same stain, lighting, camera, and magnification you expect at use time. A useful first pilot is 50–100 representative fields with all three classes; more independently collected and carefully labeled fields are better. Split by slide/patient before training so related fields cannot leak between train, validation, and test.

1. Create `personal_data/images/{train,val,test}` and matching `personal_data/labels/{train,val,test}` folders. Put your images in the appropriate image split.
2. Generate draft boxes for each split. For example:

   ```bash
   python prelabel_images.py --images personal_data/images/train --labels personal_data/labels/train --review personal_data/review/train
   python prelabel_images.py --images personal_data/images/val --labels personal_data/labels/val --review personal_data/review/val
   python prelabel_images.py --images personal_data/images/test --labels personal_data/labels/test --review personal_data/review/test
   ```

3. Manually correct every draft label. Inspect overlays in `personal_data/review/`; the model may miss cells, include debris, or assign the wrong class. Do not use unreviewed draft labels for training.
4. Fine-tune the TXL-PBC model on your reviewed images:

   ```bash
   python train_model.py --data personal_data --model runs/detect/txl_pbc_blood_cell_counter/weights/best.pt --epochs 25 --imgsz 512 --batch 16 --device mps --name microscope_finetuned
   ```

   Use `--device cpu` if MPS is unavailable, and reduce `--batch` if memory runs short. The app automatically loads the newest checkpoint under `runs/`.

Keep your test labels separate and use them only for a final evaluation. The public-dataset score does not establish performance on your microscope; report the metrics from your own held-out images.

## How it works

- The trained detector returns bounding boxes for each of the three classes.
- Before training, red blood cells are proposed with OpenCV's Hough circle detector and purple-stained connected components are heuristically divided into likely WBC and platelet candidates.
- The preview and CSV report list class-wise counts.

The controls allow adjustment of the red-cell radius range, circle sensitivity, and minimum stained area for WBC candidates. They help adapt the heuristics to different image scales and staining.

## Limitations

Training quality depends on annotation quality and how well the labeled images represent future samples. A trained detector still needs independent validation and is not clinically validated. Before training, the fallback heuristic does not reliably separate touching cells or distinguish all staining artifacts. Inspect overlays and do not use results for diagnosis or treatment decisions.

## Cloud deployment

The trained checkpoint is bundled at `models/blood_cell_counter.pt`. See [`CLOUD_DEPLOYMENT.md`](CLOUD_DEPLOYMENT.md) for Streamlit Community Cloud and Docker deployment steps.

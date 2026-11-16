# TXL-PBC blood-cell detection dataset

The active dataset in this folder is TXL-PBC: 882 training images, 252 validation images, and 126 test images, with 18,143 bounding boxes in total. It covers red blood cells, white blood cells, and platelets. The images and labels are in YOLO format.

Source: [TXL-PBC_Dataset](https://github.com/lugan113/TXL-PBC_Dataset). The source repository lists the dataset under the MIT License; a copy is in [`LICENSE-TXL-PBC.txt`](LICENSE-TXL-PBC.txt). The source `data.yaml`, annotation protocol, and metadata are also kept in this folder.

The source order is WBC=0, RBC=1, Platelets=2. To match this app's convention, the active label files have been remapped to RBC=0, WBC=1, Platelets=2. The original BCCD files are preserved separately in [`bccd_original/`](bccd_original/) and are not included in the active train/validation/test split; TXL-PBC already incorporates selected BCCD data.

## Folder layout

```text
dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
```

Each label file contains one object per line as `class_id x_center y_center width height`, with normalized coordinates from 0 to 1. Keep the official TXL-PBC split for comparisons and do not combine it with the archived BCCD images, since those sources overlap.

TXL-PBC is much larger than BCCD but is still a limited research dataset. Check the annotations and assess the model on independent images from your microscope and staining workflow. It is not a clinical diagnostic dataset.

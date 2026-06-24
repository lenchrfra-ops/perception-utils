[README_perception-utils.md](https://github.com/user-attachments/files/29284042/README_perception-utils.md)
# perception-utils

A small collection of core computer vision utility functions commonly used in object detection and perception pipelines, implemented from scratch in PyTorch.

## Why this repo exists

These functions (IoU, NMS, accuracy) are available in libraries like `torchvision.ops`, but I implemented them from scratch to deepen my own understanding of the underlying logic — and to have clean, well-tested reference implementations I can reuse and explain in detail.

## What's included

- **`compute_iou`** — Intersection over Union between two bounding boxes (single pair)
- **`compute_iou_vectorized`** — IoU computation across a batch, fully vectorized (no Python loops)
- **`nms`** — Non-Maximum Suppression to remove duplicate/overlapping detections
- **`accuracy`** — Classification accuracy from raw model outputs and ground-truth labels

## Usage

```python
from perception_utils import compute_iou, nms, accuracy

# IoU between two boxes
iou = compute_iou([10, 10, 50, 50], [30, 30, 70, 70])

# NMS on a set of detections
boxes = [[10, 10, 50, 50], [12, 12, 52, 52], [200, 200, 250, 250]]
confidences = [0.95, 0.80, 0.90]
kept_indices = nms(boxes, confidences, iou_threshold=0.5)
```

## Running the tests

```bash
pip install torch
python perception_utils.py
```

This runs a quick smoke test for each function with known expected outputs.

## Background

These implementations grew out of interview preparation for computer vision / perception engineering roles, where I wanted to make sure I deeply understood the math and logic behind standard CV operations — not just how to call a library function.

## License

MIT

"""
A small collection of core computer vision utility functions commonly used
in object detection pipelines: IoU computation (single-pair and vectorized),
Non-Maximum Suppression (NMS), and basic classification accuracy.

These are written from scratch (no torchvision.ops dependency) to demonstrate
understanding of the underlying logic, not just library usage.
"""

import torch


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a, box_b: [x1, y1, x2, y2] format (top-left, bottom-right corners)

    Returns:
        float: IoU score between 0.0 (no overlap) and 1.0 (perfect overlap)
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_vectorized(pred_boxes, gt_boxes):
    """
    Compute IoU between corresponding pairs of boxes in a batch, fully
    vectorized with PyTorch (no Python loops).

    Args:
        pred_boxes: Tensor of shape [N, 4], format [x1, y1, x2, y2]
        gt_boxes:   Tensor of shape [N, 4], format [x1, y1, x2, y2]

    Returns:
        Tensor of shape [N]: IoU for each corresponding pair
    """
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    intersection = inter_w * inter_h

    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_pred + area_gt - intersection

    return intersection / union


def nms(boxes, confidences, iou_threshold=0.5):
    """
    Non-Maximum Suppression: removes duplicate/overlapping detections,
    keeping only the highest-confidence box for each cluster of overlaps.

    Args:
        boxes: list of [x1, y1, x2, y2] boxes
        confidences: list of confidence scores, same length as boxes
        iou_threshold: boxes with IoU above this are considered duplicates

    Returns:
        list[int]: indices of boxes to keep
    """
    indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    kept = []
    while indices:
        best = indices[0]
        kept.append(best)

        remaining = []
        for i in indices[1:]:
            iou = compute_iou(boxes[best], boxes[i])
            if iou < iou_threshold:
                remaining.append(i)

        indices = remaining

    return kept


def accuracy(outputs, labels):
    """
    Compute classification accuracy.

    Args:
        outputs: Tensor of shape [B, num_classes] (raw model scores)
        labels: Tensor of shape [B] (ground-truth class indices)

    Returns:
        float: fraction of correct predictions (0.0 to 1.0)
    """
    predicted = outputs.argmax(dim=1)
    correct = (predicted == labels).sum()
    return (correct / len(labels)).item()


if __name__ == "__main__":
    # Quick smoke test / usage example
    box_a = [10, 10, 50, 50]
    box_b = [30, 30, 70, 70]
    print(f"IoU (single pair): {compute_iou(box_a, box_b):.3f}")

    boxes = [
        [10, 10, 50, 50],
        [12, 12, 52, 52],
        [11, 11, 51, 51],
        [200, 200, 250, 250],
    ]
    confidences = [0.95, 0.80, 0.85, 0.90]
    kept = nms(boxes, confidences, iou_threshold=0.5)
    print(f"NMS kept indices: {kept}  (expected: [0, 3])")

    outputs = torch.tensor([
        [0.1, 0.9, 0.2, 0.1, 0.1],
        [0.8, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.9, 0.1],
    ])
    labels = torch.tensor([1, 0, 2])
    print(f"Accuracy: {accuracy(outputs, labels):.3f}  (expected: 0.667)")

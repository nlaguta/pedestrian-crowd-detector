"""
Анализ ошибок модели на val/test: сохраняет изображения с GT и предсказаниями.

Красные рамки — предсказания модели.
Зелёные рамки — истинная разметка.

Пример:
python src/analyze_errors.py --weights runs/detect/yolov8n/weights/best.pt --data data/data.yaml --out report/error_examples
"""

import argparse
from pathlib import Path
import yaml

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
from tqdm import tqdm


def load_model(weights: str):
    if "rtdetr" in weights.lower():
        return RTDETR(weights)
    return YOLO(weights)


def resolve_dataset_paths(data_yaml: Path, split="val"):
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", "."))
    images_rel = data.get(split) or data.get("val")
    images_dir = Path(images_rel)
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    return images_dir, labels_dir


def read_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts)
        if int(cls) != 0:
            continue
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=float)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = np.maximum(0, ax2 - ax1) * np.maximum(0, ay2 - ay1)
    area_b = np.maximum(0, bx2 - bx1) * np.maximum(0, by2 - by1)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match_stats(gt, pred, iou_thr=0.5):
    if len(gt) == 0:
        return 0, 0, len(pred)
    if len(pred) == 0:
        return 0, len(gt), 0
    ious = iou_matrix(gt, pred)
    matched_gt = set()
    matched_pr = set()
    while True:
        idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[idx] < iou_thr:
            break
        gi, pi = int(idx[0]), int(idx[1])
        matched_gt.add(gi)
        matched_pr.add(pi)
        ious[gi, :] = -1
        ious[:, pi] = -1
    tp = len(matched_gt)
    fn = len(gt) - tp
    fp = len(pred) - len(matched_pr)
    return tp, fn, fp


def draw(img, gt, pred, confs):
    for box in gt:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(img, "GT", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 2)
    for box, conf in zip(pred, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"P {conf:.2f}", (x1, min(img.shape[0]-5, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="report/error_examples")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()

    model = load_model(args.weights)
    images_dir, labels_dir = resolve_dataset_paths(Path(args.data), args.split)
    out_dir = Path(args.out)
    success_dir = out_dir / "success"
    error_dir = out_dir / "errors"
    success_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(images_dir.glob(ext))
    image_paths = sorted(image_paths)[: args.max_images]

    saved_success = 0
    saved_errors = 0

    for img_path in tqdm(image_paths, desc="analyze"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = read_yolo_labels(labels_dir / f"{img_path.stem}.txt", w, h)

        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        pred = []
        confs = []
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls[0].item()) == 0:
                    pred.append(b.xyxy[0].cpu().numpy())
                    confs.append(float(b.conf[0].item()))
        pred = np.array(pred, dtype=float)

        tp, fn, fp = match_stats(gt, pred, iou_thr=0.5)
        rendered = draw(img.copy(), gt, pred, confs)

        if fn == 0 and fp <= 1 and saved_success < 3:
            cv2.imwrite(str(success_dir / f"{img_path.stem}_success.jpg"), rendered)
            saved_success += 1
        elif (fn > 0 or fp > 0) and saved_errors < 3:
            cv2.imwrite(str(error_dir / f"{img_path.stem}_error_fn{fn}_fp{fp}.jpg"), rendered)
            saved_errors += 1

        if saved_success >= 3 and saved_errors >= 3:
            break

    print(f"Сохранено успешных примеров: {saved_success}")
    print(f"Сохранено ошибочных примеров: {saved_errors}")
    print(f"Папка: {out_dir}")


if __name__ == "__main__":
    main()

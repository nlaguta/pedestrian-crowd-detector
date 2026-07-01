"""
Обучение/дообучение нескольких архитектур через Ultralytics.

Пример:
python src/train_ultralytics.py --data data/data.yaml --epochs 30 --imgsz 640 --batch 8

Для слабого ПК:
python src/train_ultralytics.py --data data/data.yaml --epochs 3 --imgsz 416 --batch 4
"""

import argparse
import csv
from pathlib import Path
import time

from ultralytics import YOLO, RTDETR


DEFAULT_MODELS = [
    "yolov5nu.pt",
    "yolov8n.pt",
    "yolov9t.pt",
    "yolov10n.pt",
    "rtdetr-l.pt",
]


def load_model(weights_name: str):
    if weights_name.lower().startswith("rtdetr"):
        return RTDETR(weights_name)
    return YOLO(weights_name)


def safe_metrics(metrics):
    """Достаём основные метрики Ultralytics. API может немного отличаться между версиями."""
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}

    def get(obj, name, default=None):
        try:
            return getattr(obj, name)
        except Exception:
            return default

    precision = get(box, "mp", None)
    recall = get(box, "mr", None)
    map50 = get(box, "map50", None)
    map5095 = get(box, "map", None)

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map5095,
        "preprocess_ms": speed.get("preprocess"),
        "inference_ms": speed.get("inference"),
        "postprocess_ms": speed.get("postprocess"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.yaml", help="Путь к data.yaml")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Список моделей")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="0 для GPU, cpu для CPU; по умолчанию auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--out", default="report/metrics_after_train.csv")
    args = parser.parse_args()

    rows = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        run_name = Path(model_name).stem.replace(".", "_")
        print(f"\n=== Обучение {model_name} ===")
        start = time.time()
        model = load_model(model_name)

        train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=run_name,
            exist_ok=True,
            seed=42,
            patience=10,
        )
        if args.device is not None:
            train_kwargs["device"] = args.device

        model.train(**train_kwargs)

        print(f"\n=== Валидация {model_name} ===")
        metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)
        row = {
            "model": model_name,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "train_time_min": round((time.time() - start) / 60, 2),
        }
        row.update(safe_metrics(metrics))
        rows.append(row)

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nГотово. Таблица метрик сохранена: {args.out}")


if __name__ == "__main__":
    main()

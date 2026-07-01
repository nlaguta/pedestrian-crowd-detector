"""
Отдельная валидация уже обученных моделей.

Скрипт ищет файлы best.pt внутри runs/detect/**/weights/best.pt
и считает метрики на одном и том же val/test наборе.

Пример:
python src/evaluate_ultralytics.py --data data/data.yaml --weights runs/detect --out report/metrics.csv
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO, RTDETR


def load_model(weights: str):
    name = Path(weights).name.lower()
    parent = str(Path(weights).parent.parent).lower()
    if "rtdetr" in name or "rtdetr" in parent:
        return RTDETR(weights)
    return YOLO(weights)


def safe_metrics(metrics):
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}

    def g(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    precision = g(box, "mp")
    recall = g(box, "mr")
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": g(box, "map50"),
        "map50_95": g(box, "map"),
        "preprocess_ms": speed.get("preprocess"),
        "inference_ms": speed.get("inference"),
        "postprocess_ms": speed.get("postprocess"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--weights", default="runs/detect", help="Папка с экспериментами или конкретный best.pt")
    parser.add_argument("--out", default="report/metrics.csv")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    wpath = Path(args.weights)
    if wpath.is_file():
        weights = [wpath]
    else:
        weights = sorted(wpath.glob("**/weights/best.pt"))

    if not weights:
        raise FileNotFoundError("Не найдено best.pt. Сначала запустите обучение или укажите путь через --weights.")

    rows = []
    for w in weights:
        print(f"Валидация: {w}")
        model = load_model(str(w))
        metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)
        row = {"model": w.parent.parent.name, "weights": str(w)}
        row.update(safe_metrics(metrics))
        rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Таблица сохранена: {args.out}")


if __name__ == "__main__":
    main()

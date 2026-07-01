"""
Демонстрационный модуль: находит людей и выводит количество найденных объектов.

Пример:
python src/demo_count_people.py --weights runs/detect/yolov8n/weights/best.pt --source demo/input --out demo/output
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO, RTDETR


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def load_model(weights: str):
    if "rtdetr" in weights.lower():
        return RTDETR(weights)
    return YOLO(weights)


def draw_boxes(frame, boxes, confs):
    count = 0
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(frame, f"People count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4)
    cv2.putText(frame, f"People count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
    return frame, count


def predict_frame(model, frame, conf=0.25, imgsz=640):
    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    r = results[0]
    boxes, confs = [], []
    if r.boxes is not None:
        for b in r.boxes:
            cls = int(b.cls[0].item())
            if cls != 0:
                continue
            boxes.append(b.xyxy[0].cpu().numpy())
            confs.append(float(b.conf[0].item()))
    return boxes, confs


def process_image(model, path: Path, out_dir: Path, conf: float, imgsz: int):
    img = cv2.imread(str(path))
    if img is None:
        print(f"Не удалось открыть: {path}")
        return
    boxes, confs = predict_frame(model, img, conf, imgsz)
    img, count = draw_boxes(img, boxes, confs)
    out_path = out_dir / f"{path.stem}_detected{path.suffix}"
    cv2.imwrite(str(out_path), img)
    print(f"{path.name}: найдено людей = {count}; файл: {out_path}")


def process_video(model, path: Path, out_dir: Path, conf: float, imgsz: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{path.stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    counts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, confs = predict_frame(model, frame, conf, imgsz)
        frame, count = draw_boxes(frame, boxes, confs)
        counts.append(count)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    avg_count = sum(counts) / len(counts) if counts else 0
    print(f"{path.name}: среднее количество людей = {avg_count:.1f}; файл: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="demo/input")
    parser.add_argument("--out", default="demo/output")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = load_model(args.weights)
    source = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [source] if source.is_file() else sorted([p for p in source.iterdir() if p.is_file()])
    for p in files:
        if p.suffix.lower() in IMAGE_EXTS:
            process_image(model, p, out_dir, args.conf, args.imgsz)
        elif p.suffix.lower() in VIDEO_EXTS:
            process_video(model, p, out_dir, args.conf, args.imgsz)


if __name__ == "__main__":
    main()

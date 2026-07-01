"""
Конвертация оригинального CrowdHuman .odgt в YOLO format.

Ожидается:
- annotation_train.odgt / annotation_val.odgt
- папки изображений, где имя файла обычно совпадает с ID из разметки: <ID>.jpg

YOLO label:
<class_id> <x_center> <y_center> <width> <height>
координаты нормированы от 0 до 1.

Для варианта 14 лучше использовать full-body box: fbox.
Можно поменять --box-type на vbox, если нужна видимая часть человека.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
import yaml


def yolo_line_from_box(box, img_w, img_h, class_id=0):
    x, y, w, h = box
    x1 = max(0, float(x))
    y1 = max(0, float(y))
    x2 = min(img_w, float(x) + float(w))
    y2 = min(img_h, float(y) + float(h))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw <= 1 or bh <= 1:
        return None

    xc = x1 + bw / 2
    yc = y1 + bh / 2

    return f"{class_id} {xc/img_w:.6f} {yc/img_h:.6f} {bw/img_w:.6f} {bh/img_h:.6f}"


def convert_split(images_dir: Path, ann_path: Path, out_dir: Path, split: str, box_type: str, copy_images: bool):
    img_out = out_dir / "images" / split
    lab_out = out_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    with ann_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    converted_images = 0
    converted_boxes = 0
    skipped = 0

    for line in tqdm(lines, desc=f"convert {split}"):
        item = json.loads(line)
        image_id = item["ID"]

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        label_lines = []

        for gt in item.get("gtboxes", []):
            extra = gt.get("extra", {})
            if int(extra.get("ignore", 0)) == 1:
                continue

            box = gt.get(box_type)
            if box is None:
                continue

            yline = yolo_line_from_box(box, w, h)
            if yline:
                label_lines.append(yline)

        if not label_lines:
            skipped += 1
            continue

        if copy_images:
            shutil.copy2(img_path, img_out / img_path.name)
        else:
            # создаём относительную копию через hardlink, если ОС разрешит, иначе обычная копия
            try:
                target = img_out / img_path.name
                if not target.exists():
                    target.hardlink_to(img_path)
            except Exception:
                shutil.copy2(img_path, img_out / img_path.name)

        (lab_out / f"{img_path.stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        converted_images += 1
        converted_boxes += len(label_lines)

    print(f"{split}: images={converted_images}, boxes={converted_boxes}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-train", required=True, help="Папка с train изображениями CrowdHuman")
    parser.add_argument("--images-val", required=True, help="Папка с val изображениями CrowdHuman")
    parser.add_argument("--ann-train", required=True, help="annotation_train.odgt")
    parser.add_argument("--ann-val", required=True, help="annotation_val.odgt")
    parser.add_argument("--out", required=True, help="Выходная папка YOLO датасета")
    parser.add_argument("--box-type", default="fbox", choices=["fbox", "vbox", "hbox"], help="Тип рамки: fbox — всё тело, vbox — видимая часть, hbox — голова")
    parser.add_argument("--copy-images", action="store_true", help="Копировать изображения вместо hardlink")
    args = parser.parse_args()

    out_dir = Path(args.out)
    convert_split(Path(args.images_train), Path(args.ann_train), out_dir, "train", args.box_type, args.copy_images)
    convert_split(Path(args.images_val), Path(args.ann_val), out_dir, "val", args.box_type, args.copy_images)

    data = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": {0: "person"},
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"Готово. data.yaml создан: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

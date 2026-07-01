# Вариант 14 — Детекция пешеходов в плотной толпе

Практическая работа по компьютерному зрению: нужно разработать прототип детектора людей в сценах с перекрытиями, сравнить не менее 5 архитектур, выбрать лучшую модель и показать ошибки на сложных примерах.

## Что делает проект

- обучает / дообучает модели детекции на датасете с людьми;
- сравнивает 5 архитектур в одинаковых условиях;
- считает метрики качества и скорость инференса;
- строит демонстрационный модуль: изображение или видео → рамки найденных людей → количество людей;
- сохраняет примеры успешной работы и ошибок для отчёта.

## Рекомендуемые архитектуры для сравнения

В работе нельзя считать разными архитектурами только разные размеры одной модели, например `yolov8n`, `yolov8s`, `yolov8m`. Поэтому в проекте выбраны разные семейства/подходы:

1. `YOLOv5u` — быстрый one-stage detector.
2. `YOLOv8` — современный one-stage detector.
3. `YOLOv9` — YOLO-семейство с другой архитектурной идеей.
4. `YOLOv10` или `YOLO11` — современная версия YOLO для сравнения скорости и качества.
5. `RT-DETR` — transformer-based real-time detector.

Если какая-то модель не скачивается в вашей версии Ultralytics, обновите пакет:

```bash
pip install -U ultralytics
```

или замените модель на любую поддерживаемую из официального списка Ultralytics.

## Структура проекта

```text
project/
  data/
    data.yaml                       # описание датасета YOLO
  demo/
    input/                          # сюда положить тестовые изображения/видео
    output/                         # сюда сохраняются результаты
  report/
    report_template.md              # текстовые блоки для отчёта
    metrics_template.csv            # шаблон таблицы сравнения
  src/
    convert_crowdhuman_to_yolo.py   # конвертация CrowdHuman в YOLO-разметку
    train_ultralytics.py            # обучение 5 моделей
    evaluate_ultralytics.py         # отдельная валидация и таблица метрик
    demo_count_people.py            # демонстрация: детекция и подсчёт людей
    analyze_errors.py               # поиск успешных и ошибочных примеров
  requirements.txt
  README.md
```

## Быстрый запуск

### 1. Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

На macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Подготовка датасета

Самый простой вариант — скачать датасет в формате YOLO с Roboflow/Kaggle/другого источника и положить так:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

После этого исправьте пути в `data/data.yaml`.

Если используете оригинальный CrowdHuman, запустите конвертацию:

```bash
python src/convert_crowdhuman_to_yolo.py ^
  --images-train "D:\datasets\CrowdHuman\Images_train" ^
  --images-val "D:\datasets\CrowdHuman\Images_val" ^
  --ann-train "D:\datasets\CrowdHuman\annotation_train.odgt" ^
  --ann-val "D:\datasets\CrowdHuman\annotation_val.odgt" ^
  --out "D:\datasets\crowdhuman_yolo"
```

### 3. Обучение 5 архитектур

```bash
python src/train_ultralytics.py --data data/data.yaml --epochs 30 --imgsz 640 --batch 8
```

Для слабого ноутбука можно начать с проверки:

```bash
python src/train_ultralytics.py --data data/data.yaml --epochs 3 --imgsz 416 --batch 4
```

### 4. Сравнение моделей

```bash
python src/evaluate_ultralytics.py --data data/data.yaml --weights runs/detect --out report/metrics.csv
```

### 5. Демонстрация подсчёта людей

Положите фото или видео в `demo/input`, затем:

```bash
python src/demo_count_people.py --weights runs/detect/yolov8n/weights/best.pt --source demo/input --out demo/output
```

### 6. Анализ ошибок

```bash
python src/analyze_errors.py --weights runs/detect/yolov8n/weights/best.pt --data data/data.yaml --out report/error_examples
```

## Что загрузить на GitHub

Загружайте:

- `README.md`
- `requirements.txt`
- папку `src`
- папку `data` с `data.yaml`
- папку `report`
- несколько примеров из `demo/output`
- итоговую таблицу `report/metrics.csv`

Не загружайте большие датасеты, веса моделей и папку `.venv`. Они исключены в `.gitignore`.

## Команды для GitHub

```bash
git init
git add .
git commit -m "Variant 14 pedestrian detection in dense crowd"
git branch -M main
git remote add origin https://github.com/YOUR_LOGIN/pedestrian-crowd-detector.git
git push -u origin main
```

После загрузки ссылка для преподавателя будет примерно такая:

```text
https://github.com/YOUR_LOGIN/pedestrian-crowd-detector
```

## Краткий вывод для отчёта

В работе реализован прототип детектора пешеходов в плотной толпе. Для сравнения выбраны пять архитектур детекции, которые обучались и оценивались на одинаковом наборе данных. Качество сравнивалось по mAP@0.5, mAP@0.5:0.95, precision, recall, F1, а практическая применимость — по скорости инференса и количеству ошибок в сценах с перекрытиями. Лучшей считается модель, которая даёт не только высокую метрику, но и устойчиво находит частично закрытых людей.
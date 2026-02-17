
# Классификация типов землепользования по спутниковым снимкам (EuroSAT)

Автор: **Галкин Дмитрий Михайлович**

Проект: обучение модели компьютерного зрения для определения типа землепользования по спутниковым RGB-снимкам на датасете **EuroSAT**. Репозиторий оформлен как воспроизводимый ML-пайплайн: управление данными через DVC, конфигурация через Hydra, обучение через PyTorch Lightning, логирование через MLflow, экспорт в ONNX и упаковка модели для сервинга через MLflow Serving.

---

## Постановка задачи

Цель — обучить модель, которая по спутниковому RGB-изображению предсказывает класс землепользования/земного покрова.  
Задача формализуется как мультиклассовая классификация на 10 классов.

---

## Формат входных и выходных данных

### Обучение/валидация/тест
Вход: изображение `jpg` (RGB).  
Внутренний формат: `torch.Tensor float32` формы `(3, H, W)`.  
Размер входа:
- для ResNet18: `224x224`
- для baseline CNN: `64x64`

Выход: логиты `torch.Tensor` формы `(batch, 10)`.

### Инференс
Вход: путь к изображению (абсолютный или относительный).  
Выход: JSON с полями:
- `predicted_class`
- `probabilities`
- `topk`

---

## Метрики

- Accuracy
- Macro F1-score
- Confusion matrix

---

## Валидация и воспроизводимость

Разбиение датасета стратифицированное:
- train: 70%
- val: 15%
- test: 15%

Воспроизводимость:
- фиксированный seed в конфиге
- детерминизм torch в `eurosat_landuse/utils/repro.py`

---

## Данные

### Датасет
Используется EuroSAT RGB (структура `ImageFolder`).  
10 классов: `AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake`.

Источник загрузки указан в `configs/data/eurosat.yaml`.

### Управление данными (DVC)
Данные не хранятся в git.  
В git хранятся DVC-метафайлы:
- `.dvc/`, `.dvcignore`
- `data/raw/EuroSAT_RGB.dvc`

Загрузка данных встроена в команды `train` и `infer`.

---

## Моделирование

Baseline: простая CNN `SimpleCNN` (`eurosat_landuse/models/simple_cnn.py`).  
Основная модель: ResNet18 (`torchvision.models`) с fine-tuning под 10 классов (`eurosat_landuse/models/factory.py`).

---

## Внедрение

Использование модели:
- CLI-инференс через команду `infer`
- сервер инференса через MLflow Serving на основе `artifacts/mlflow_model`

---

# Техническая часть

## Требования к окружению

Ограничение версии Python задано в `pyproject.toml`:
- Python >= 3.10 и < 3.13

Управление зависимостями: Poetry.  
Фиксация версий: `poetry.lock`.

---

## Setup

Установка зависимостей:
```bash
poetry install
```

Установка и прогон хуков качества:
```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

---

## MLflow

Запуск MLflow tracking server:
```bash
poetry run mlflow server \
  --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

---

## Data

Скачивание данных:
```bash
poetry run python -m eurosat_landuse.commands download_data
```

DVC:
```bash
poetry run dvc init
poetry run dvc add data/raw/EuroSAT_RGB
```

Локальное хранилище DVC:
```bash
mkdir -p ../dvc-storage
poetry run dvc remote add -d localstorage ../dvc-storage
poetry run dvc push
```

---

## Train

Обучение ResNet18:
```bash
poetry run python -m eurosat_landuse.commands train
```

Переопределение параметров:
```bash
poetry run python -m eurosat_landuse.commands train trainer.max_epochs=5 train.batch_size=128 train.lr=1e-4
```

Baseline CNN:
```bash
poetry run python -m eurosat_landuse.commands train model=simple_cnn model.image_size=64 train.lr=1e-3
```

Результаты:
- `artifacts/best.ckpt`
- `artifacts/model_weights.pth`
- `plots/loss_curves.png`
- `plots/val_acc.png`
- `plots/confusion_matrix.png`

---

## Production preparation

Экспорт ONNX:
```bash
poetry run python -m eurosat_landuse.commands export_onnx infer.checkpoint_path=artifacts/best.ckpt
```

Экспорт MLflow model:
```bash
poetry run python -m eurosat_landuse.commands export_mlflow_model
```

---

## Infer

CLI-инференс:
```bash
poetry run python -m eurosat_landuse.commands infer infer.image_path=/ABS/PATH/TO/IMAGE.jpg
```

CLI-инференс через MLflow model:
```bash
poetry run python -m eurosat_landuse.commands infer_mlflow_model \
  --image_path /ABS/PATH/TO/IMAGE.jpg \
  --model_path artifacts/mlflow_model
```

---

## Inference server (MLflow Serving)

Поднятие сервера:
```bash
poetry run mlflow models serve -m artifacts/mlflow_model -p 5001 --no-conda
```

Запрос:
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split":{"columns":["image_path"],"data":[["/ABS/PATH/TO/IMAGE.jpg"]]}}'
```

---

## Быстрая проверка end-to-end

Терминал 1:
```bash
poetry run mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Терминал 2:
```bash
poetry run python -m eurosat_landuse.commands download_data
poetry run python -m eurosat_landuse.commands train trainer.max_epochs=2
```

Инференс на одном изображении из датасета:
```bash
IMAGE_PATH="$(find "$(pwd)/data/raw/EuroSAT_RGB" -type f -name "*.jpg" | head -n 1)"
poetry run python -m eurosat_landuse.commands infer infer.image_path="$IMAGE_PATH"
```

---

## Структура репозитория

- `eurosat_landuse/commands.py` — CLI (download/train/export/infer)
- `eurosat_landuse/data/` — загрузка данных, разбиение, LightningDataModule
- `eurosat_landuse/models/` — модели
- `eurosat_landuse/training/` — LightningModule и метрики
- `eurosat_landuse/serving/` — MLflow pyfunc model
- `configs/` — Hydra конфиги
- `plots/` — графики
- `artifacts/` — артефакты

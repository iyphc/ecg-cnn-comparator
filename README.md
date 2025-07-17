# ECG Analyzer

## Установка

### Из исходного кода (режим разработки)
```bash
git clone https://github.com/iyphc/ecg-cnn-comparator.git
cd ecg-cnn-comparator
pip install -e .
```

## Структура проекта

```
src/ecg_analyzer/
├── __init__.py              # Главный модуль пакета
├── main.py                  # CLI entry point
├── data/                    # Модуль работы с данными
│   ├── __init__.py
│   ├── loader.py           # Загрузка данных
│   └── preprocess.py       # Предобработка данных
├── models/                  # Модуль моделей
│   ├── __init__.py
│   ├── base_model.py       # Базовая модель
│   ├── resnet18.py         # ResNet для 1D сигналов
│   └── cnn_handcrafted.py  # Модель с ручными признаками
├── training/               # Модуль обучения
│   ├── __init__.py
│   ├── trainer.py          # Обучение моделей
│   └── evaluator.py        # Оценка моделей
└── utils/                  # Вспомогательные утилиты
    ├── __init__.py
    ├── constants.py        # Константы
    ├── handlers.py         # Обработчики
    └── utils.py            # Утилиты
```

## Использование

### Импорт пакета
```python
import ecg_analyzer
print(f"ECG Analyzer version: {ecg_analyzer.__version__}")
```

### Загрузка данных
```python
from ecg_analyzer import get_dataloaders

# Загрузка данных PTB-XL
train_loader, test_loader, val_loader, classes, features = get_dataloaders(
    batch_size=64,
    valid_part=0.2,
    raw_path='data/raw/physionet.org/files/ptb-xl/1.0.1/',
    sampling_rate=100
)
```

### Создание и обучение модели
```python
from ecg_analyzer import BaseModel, HandcraftedModel, train_model

# Создание модели
model = BaseModel(in_channels=12, out_classes=len(classes))
# или
model = BaseModel(in_channels=12, out_classes=len(classes))
model = HandcraftedModel(base_model=model, handcrafted_classes=len(features))

# Обучение модели
train_model(
    model=model,
    train_load=train_loader,
    test_load=test_loader,
    val_load=val_loader,
    class_names=classes,
    epochs=10,
    learning_rate=0.001
)
```

### Оценка модели
```python
from ecg_analyzer import evaluate_model

metrics = evaluate_model(
    model=model,
    test_loader=test_loader,
    is_handcrafted=False
)
```

### Использование CLI
```bash
# Обучение модели
pytho3 -m ecg_analyzer.main mode=train

# Сравнение моделей
python3 -m ecg_analyzer.main mode=compare

# Оценка модели
python3 -m ecg_analyzer.main mode=evaluate
```

## Конфигурация

Проект использует Hydra для управления конфигурацией. Основные файлы конфигурации:

- `configs/config.yaml` - Основная конфигурация
- `configs/data.yaml` - Конфигурация данных
- `configs/training.yaml` - Конфигурация обучения
- `configs/base_model/` - Конфигурации моделей
- `configs/handcrafted_model/` - Конфигурации handcrafted оберток

## Лицензия

MIT License - см. файл LICENSE для подробностей.

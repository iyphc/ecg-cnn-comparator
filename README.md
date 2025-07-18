 # ECG Models Comparison


## Описание
Проект реализует и сравнивает различные архитектуры нейронных сетей для классификации или анализа ЭКГ-сигналов, включая:
- Сверточные нейронные сети (CNN) с "сырыми" и "ручными" признаками
- Базовые классы и утилиты для обучения и оценки моделей


### Основные цели
* [ ] **Исправить handcrafted**
* [ ] [roc-auc баг](https://github.com/iyphc/ecg-cnn-comparator/issues/7#issue-3222168220)
* [ ] Реализовать более подробный подсчет метрик
* [ ] Улучшить BaseModel 
* [ ] Дополнительно реализовать ResNet


### Побочные цели
1. Описать функции и работу
2. Оформить результаты тестов, сделать выводы
3. Переписать скрипт запуска
    - Сделать опционализацию (hydra вместо argparse)
4. Переработать стурктуру проекта (train / predict не относятся к структуре src/model)
5. setup.py 

## Структура проекта

```
ECG_models_comparison/
├── main.py                 # Главный скрипт запуска
├── src/
│   ├── data/               # Модули для работы с данными
│   │   ├── constants.py
│   │   ├── loader.py
│   │   ├── preprocess.py
│   └── models/             # Модули моделей и утилиты
│       ├── base_model.py
│       ├── cnn_handcrafted.py
│       ├── resnet.py
│       ├── evaluation.py
│       ├── train.py
│       └── utils.py
```

## Запуск

```
pip install requirements.txt
python3 main.py
```

Установить датасет нужно в папку data/raw. 

### Ссылки на устновку датасета PTB-XL:
[physionet](https://physionet.org/content/ptb-xl/1.0.1/) \
[Kaggle](https://www.kaggle.com/datasets/rohitdwivedula/ptbxl-original-dataset/data)
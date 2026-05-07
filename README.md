# IMU SSL Running Classification

Проект для классификации беговых упражнений по IMU-сигналам и сравнения обычного supervised-обучения с простыми self-supervised learning подходами.

В репозитории лежит полный учебный pipeline:

1. первичный анализ структуры JSON-датасета;
2. преобразование IMU-сигналов в окна фиксированной длины;
3. subject-wise 5-fold разбиение без утечки участников между train и validation;
4. обучение 1D-CNN baseline;
5. SSL pretraining через masked reconstruction и temporal permutation;
6. fine-tuning SSL encoder'ов на downstream-классификацию;
7. сохранение метрик по folds и итоговой таблицы mean/std.

## Что решает проект

Задача проекта - многоклассовая классификация беговых движений по данным инерциальных датчиков. Каждый пример - короткое окно временного ряда от одного IMU-сенсора. На вход модели подаются 6 каналов:

- `acc_x`, `acc_y`, `acc_z`;
- `gyro_x`, `gyro_y`, `gyro_z`.

Классы движений:

- `carioca_left`;
- `carioca_right`;
- `heel_to_butt`;
- `high_knee_run`;
- `running`;
- `sideskips_left`;
- `sideskips_right`.

Основной исследовательский вопрос: может ли простой SSL-pretraining улучшить качество классификации по сравнению с supervised baseline, если размеченный IMU-датасет небольшой.

## Структура репозитория

```text
.
├── participant_data/                         # исходные JSON-файлы по участникам
├── data/
│   └── processed/                            # подготовленные окна, folds и результаты
├── notebooks/
│   ├── 01_dataset_check_participant_structure.ipynb
│   ├── 02_make_windows_and_split_5fold_grouped.ipynb
│   └── 03_supervised_and_simple_ssl_5fold_grouped.ipynb
├── main.py                                   # минимальная точка входа проекта
├── pyproject.toml                            # зависимости проекта
├── uv.lock                                   # lock-файл uv
├── .python-version                           # Python 3.12
├── .gitignore
└── README.md
```

## Данные

Исходные данные находятся в `participant_data/`. В проекте используется 147 JSON-файлов:

- 21 папка `participant_*`;
- по 7 файлов на папку участника;
- каждый файл соответствует одному типу движения;
- каждый JSON содержит данные от 4 IMU-сенсоров.

Внутри JSON верхнеуровневая структура содержит поля:

- `datas`;
- `deviceId`;
- `startTime`;
- `endTime`;
- `label`;
- `note`.

Главное поле для ML - `datas`. В нем лежит список из 4 сенсоров. Для каждого сенсора доступны:

- `accData`;
- `gyroData`;
- `deviceMac`.

В `accData` и `gyroData` есть временные ряды по осям `xAxisData`, `yAxisData`, `zAxisData` и временные метки `timeStamp`.

В текущем MVP используется только фиксированный `sensor_idx = 0`. В исходных JSON нет явного человекочитаемого описания расположения сенсоров на теле, поэтому проект не утверждает, что этот сенсор соответствует конкретной анатомической позиции.

## Подготовка окон

Подготовка выполняется во втором ноутбуке:

`notebooks/02_make_windows_and_split_5fold_grouped.ipynb`

Параметры обработки:

- `SENSOR_IDX = 0`;
- `WINDOW_SIZE = 52`;
- `OVERLAP = 0.75`;
- `STRIDE = 13`;
- формат окна: `[channels, time] = [6, 52]`.

Частота дискретизации по timestamp оценивается примерно как 52.6 Hz, поэтому окно длиной 52 отсчета соответствует примерно 1 секунде сигнала.

После нарезки получается:

```text
X.shape = (5270, 6, 52)
```

То есть подготовлено 5270 окон.

## Разбиение на folds

Для честной оценки используется `GroupKFold(n_splits=5)`. Группировка выполняется по участникам, чтобы окна одного и того же человека не попадали одновременно в train и validation.

Особый случай:

- `participant_12` и `participant_12_2` считаются одной группой `participant_12`.

Это сделано для защиты от возможной утечки, потому что `participant_12_2` может быть второй записью того же участника.

Нормализация выполняется отдельно внутри каждого fold:

- `mean` и `std` считаются только на `X_train`;
- `X_val` нормализуется train-статистиками текущего fold.

## Модели и эксперименты

Основной эксперимент находится в:

`notebooks/03_supervised_and_simple_ssl_5fold_grouped.ipynb`

Сравниваются три подхода:

1. `supervised_from_scratch`
   - 1D-CNN/FCN классификатор обучается напрямую по размеченным окнам.

2. `masked_reconstruction_ssl_plus_finetune`
   - encoder сначала обучается восстанавливать замаскированные timestep;
   - затем encoder переносится в классификатор и дообучается на labels.

3. `temporal_permutation_ssl_plus_finetune`
   - encoder сначала обучается определять, был ли нарушен временной порядок сегментов окна;
   - затем encoder переносится в классификатор и дообучается на labels.

Архитектура encoder'а - компактная 1D-CNN/FCN:

- несколько `Conv1d` слоев;
- `BatchNorm1d`;
- `ReLU`;
- global average pooling по временной оси;
- отдельные heads для классификации, реконструкции и binary SSL-задачи.

Метрики:

- `accuracy`;
- `macro_f1`;
- `balanced_accuracy`;
- mean и std по 5 folds.

## Итоговые результаты

Файл с итоговой таблицей:

`data/processed/03_results_5fold_summary.csv`

Результаты по 5 folds:

| Model | Accuracy mean | Accuracy std | Macro-F1 mean | Macro-F1 std | Balanced accuracy mean | Balanced accuracy std |
|---|---:|---:|---:|---:|---:|---:|
| `masked_reconstruction_ssl_plus_finetune` | 0.8297 | 0.1167 | 0.8256 | 0.1233 | 0.8279 | 0.1204 |
| `supervised_from_scratch` | 0.8426 | 0.1104 | 0.8413 | 0.1132 | 0.8411 | 0.1136 |
| `temporal_permutation_ssl_plus_finetune` | 0.8553 | 0.1141 | 0.8517 | 0.1198 | 0.8539 | 0.1171 |

Лучший средний результат показал `temporal_permutation_ssl_plus_finetune`. Улучшение относительно supervised baseline небольшое, но положительное: примерно +1 процентный пункт по macro-F1.

`masked_reconstruction_ssl_plus_finetune` в текущей конфигурации не дал устойчивого выигрыша и оказался ниже supervised baseline.

## Основные выводы

- IMU-сигналы подходят для классификации беговых движений даже на компактной 1D-CNN.
- Subject-wise split важен: случайное разбиение окон могло бы завысить качество из-за попадания данных одного участника и в train, и в validation.
- Temporal permutation лучше согласуется с природой задачи, потому что беговые движения имеют выраженную временную структуру.
- Masked reconstruction учит восстанавливать локальные участки сигнала, но не обязательно выделяет признаки, полезные для разделения классов.
- Разброс метрик по folds заметный, что ожидаемо для небольшого датасета человеческих движений.

## Установка

Проект настроен под Python 3.12 и `uv`.

```bash
uv sync
```

Если `uv` не установлен, зависимости можно поставить обычным `pip` из `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter ipykernel matplotlib numpy pandas scikit-learn torch
```

## Запуск

Рекомендуемый порядок запуска ноутбуков:

1. `notebooks/01_dataset_check_participant_structure.ipynb`
   - проверка структуры датасета, labels, participants, sensors, timestamp и sampling rate.

2. `notebooks/02_make_windows_and_split_5fold_grouped.ipynb`
   - парсинг JSON, нарезка окон, создание `GroupKFold`, fold-wise нормализация, сохранение `.npz` и metadata.

3. `notebooks/03_supervised_and_simple_ssl_5fold_grouped.ipynb`
   - обучение supervised baseline и двух SSL-подходов, расчет метрик, сохранение результатов.

Запуск Jupyter:

```bash
uv run jupyter notebook
```

или:

```bash
jupyter notebook
```

## Сохраненные артефакты

В `data/processed/` уже сохранены готовые результаты обработки:

- `sensor0_1s_overlap75_all_windows.npz` - все окна без split;
- `sensor0_1s_overlap75_fold0.npz` ... `sensor0_1s_overlap75_fold4.npz` - train/validation folds;
- `sensor0_1s_overlap75_window_meta.csv` - metadata окон;
- `sensor0_1s_overlap75_label_mapping.json` - mapping labels/id;
- `03_results_5fold_by_fold.csv` - метрики по каждому fold;
- `03_results_5fold_summary.csv` - итоговые mean/std метрики.

Файлы `03_results_by_fold.csv`, `03_results_summary.csv` и `03_simple_results_fold0.csv` относятся к более ранним/промежуточным запускам эксперимента и оставлены для воспроизводимости истории работы.

## Ограничения текущей версии

- используется только один сенсор `sensor_idx = 0`;
- расположение сенсоров на теле явно не размечено в JSON;
- нет отдельной финальной test-выборки;
- эксперименты выполнены для одного seed;
- не подбирались window size, overlap и гиперпараметры;
- не использовались sensor fusion и более сложные SSL-задачи.

## Возможные улучшения

- определить соответствие `sensor_idx` реальным позициям сенсоров;
- использовать все 4 сенсора вместо одного;
- добавить отдельный held-out test set;
- повторить эксперименты на нескольких seeds;
- сравнить 1D-CNN с LSTM/GRU, Transformer encoder или InceptionTime;
- попробовать дополнительные SSL-задачи: arrow-of-time, contrastive learning, time warping, jittering/augmentation prediction;
- вынести код из ноутбуков в Python-модули и добавить CLI для воспроизводимого запуска.

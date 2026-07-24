# Безопасная газовая плита с компьютерным зрением

## Что это?

Система компьютерного зрения для обеспечения безопасности на кухне. Проект объединяет два ключевых модуля: автоматическое распознавание показаний газового счетчика и мониторинг присутствия человека для предотвращения аварийных ситуаций.

### Ключевые возможности:
- **Автоматическое снятие показаний** газового счетчика с точностью 99.81 %
- **Мониторинг работы газа** в реальном времени
- **Отслеживание присутствия человека** на кухне
- **Оповещения** через Telegram-бота

## Как это работает

### 1. Контроль газа
Камера направлена на газовый счетчик и определяет:
- **Есть ли изменения** на счетчике (горит ли газ в данный момент);
- Если газ включен, система засекает время его работы.

### 2. Контроль присутствия человека
Вторая камера отслеживает пространство кухни:
- Фиксирует, был ли человек в кадре за последние 10 минут;
- Распознавание присутствия осуществляется средствами компьютерного зрения.

### 3. Оповещение
Если газ включен **И** человека нет на кухне дольше 10 минут →  
📨 **Telegram-бот отправляет предупреждение**

## Схема работы

```
Камера счетчика → Видит изменения → Газ включен
                                    ↓
Камера кухни → Нет человека 10 мин → ОПАСНО!
                                    ↓
                        Telegram-сообщение
```

## 📁 Структура проекта

```
meter-watch/
├── config/                              # Configuration files
│   └── grafana/                         # Grafana dashboards & provisioning
│
│
├── docker_base/                         # Base Docker images
│   ├── Dockerfile
│   └── base-requirements.txt
│
├── services/                            # Microservices
│   ├── person-detector/                 # Person detection service
│   │   ├── app/
│   │   │   ├── api.py                   # REST API endpoints
│   │   │   ├── person_tracker.py        # Core tracking logic
│   │   │   ├── video_buffer.py          # Video stream management
│   │   │   ├── telegram_bot.py          # Telegram notifications
│   │   │   ├── database.py              # DB operations
│   │   │   ├── rate_limiter.py          # Rate limiting
│   │   │   ├── safety_monitor.py        # Safety checks
│   │   │   └── state_manager.py         # State management
│   │   │ 
│   │   ├── run.py                       # Entry point
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── yolov8n.pt                   # YOLO model weights
│   │
│   └── cnn-recognizer/                  # Digit recognition service
│       ├── models/                      # ML models
│       │   ├── digit_recognizer.pth     # Trained model weights
│       │   ├── digit_recognizer.py      # Model architecture
│       │   ├── pytorch_model.py         # PyTorch wrapper
│       │   ├── error_models.py          # Error analysis
│       │   └── monitoring_models.py     # Monitoring models
│       ├── routes/                      # API routes
│       │   ├── main_routes.py
│       │   ├── manual_recognize.py
│       │   └── config_routes.py
│       ├── services/                    # Business logic
│       │   ├── recognition.py           # Main recognition logic
│       │   ├── database.py              # DB operations
│       │   ├── monitoring.py            # Metrics collection
│       │   └── config.py                # Service config
│       ├── utils/                       # Utilities
│       │   ├── api_utils.py
│       │   ├── image_utils.py
│       │   ├── preprocessing.py         # Image preprocessing
│       │   ├── augmentation.py          # Data augmentation
│       │   ├── heatmap.py               # Visualization
│       │   ├── log_data.py              # Logging
│       │   ├── number_utils.py          # Number processing
│       │   └── splitter.py              # Dataset splitting
│       │
│       ├── dataset/                     # Training datasets
│       ├── app.py                       # Flask application
│       ├── trainer.py                   # Training pipeline
│       ├── test_on_raw.py               # Testing script
│       ├── analyze_errors_top.py        # Top errors analysis
│       ├── dataset_make.py              # Dataset creation
│       ├── classification_report.txt    # Model performance report
│       ├── requirements.txt
│       ├── Dockerfile
│       │
├── experiments/                         # Research & experiments
│   ├── hog_svm/                         # HOG + SVM experiments
│   └── tracker/                         # Tracker experiments
│
├── meter-watch-shared/                  # Shared Python package
│
├── scripts/                             # Utility scripts
│   ├── balance_dataset.py               # Dataset balancing
│   └── emulate_gas_counter.py           # Gas counter emulator
│
├── notebooks/                           # Jupyter notebooks
│   └── 01_eda.ipynb                     # Exploratory data analysis
│
├── doc/                                 # Documentation
│
├── readme_images/                       # README images
│
├── alembic/                             # Database migrations
│
├── docker-compose.yml                   # Main Docker Compose
├── deploy.sh                            # Deployment script
├── alembic.ini                          # Alembic configuration
├── .env.example                                 # Environment variables
├── README.md                            # Project overview
├── RUN.md                               # Running instructions
├── TODO.md                              # Todo list
├── LICENSE                              # License
```

---

### Архитектура нейросети
```
┌─────────────────────────────────────┐
│  Input (28×28 grayscale)            │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Conv2d(1→32, 3×3, padding=1)       │
│  BatchNorm2d(32)                    │
│  ReLU                               │
│  MaxPool2d(2×2)                     │  ← Output: 14×14
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Conv2d(32→64, 3×3, padding=1)      │
│  BatchNorm2d(64)                    │
│  ReLU                               │
│  MaxPool2d(2×2)                     │  ← Output: 7×7
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Flatten: 7×7×64 = 3136             │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Linear(3136→128)                   │
│  BatchNorm1d(128)                   │
│  ReLU                               │
│  Dropout(0.3)                       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  Linear(128→10)                     │
│  Output (10 digit classes)          │
└─────────────────────────────────────┘
```

## Полный цикл работы системы

### Сценарий 1: Нормальная работа
```
1. PersonTracker обнаруживает человека → записывает last_seen в Redis
2. Газовый датчик неактивен (gas_flow = '0' или отсутствует)
3. SafetyMonitor проверяет: газ не идет → тревога не создается
4. Система находится в режиме ожидания
```

### Сценарий 2: Обнаружение утечки газа
```
1. Газовый датчик срабатывает → API устанавливает gas_flow = '1' на 5 минут
2. PersonTracker не видит человека дольше 5 минут (last_seen устаревает)
3. SafetyMonitor проверяет каждые 10 секунд:
   ✅ gas_flow == '1'
   ✅ last_seen > 300 секунд
   ✅ alert_triggered отсутствует
   ✅ startup_mode отсутствует
4. Создается alert_triggered = '1'
5. Отправляется сообщение в Telegram
```

### Сценарий 3: Человек вернулся
```
1. PersonTracker снова видит человека → обновляет last_seen
2. SafetyMonitor проверяет:
   ✅ last_seen обновлен (менее 300 сек)
   ❌ условие "нет человека" больше не выполняется
3. Alert НЕ отправляется (или сбрасывается?)
   (В коде нет автоматического сброса alert при появлении человека!)
```

### Сценарий 4: Отключение уведомлений (Cooldown)
```
1. Пользователь отправляет команду /silence в Telegram
2. Устанавливается alert_cooldown на 10 минут
3. SafetyMonitor видит активный cooldown и временно прекращает отправку новых уведомлений
4. Через 10 минут cooldown автоматически удаляется (TTL)
5. Если опасность (газ горит + никого нет) сохраняется, система возобновляет оповещения
```

### Сценарий 5: Режим запуска (первые 5 минут)
```
1. Система запускается → устанавливается ключ startup на 5 минут
2. RedisManager.set_timestamp_key(
    config.REDIS_KEYS['startup'], 
    config.STARTUP_DURATION  # 300 секунд
)
3. В SafetyMonitor:
   if RedisManager.key_exists(config.REDIS_KEYS['startup']):
       return  # НЕ проверять и НЕ отправлять тревоги
4. Через 5 минут ключ автоматически удаляется
```

# Этапы разработки

## Камера для счетчика и сбор данных

### 📷 Конструкция камеры
- **ESP32-CAM** с дополнительной **светодиодной подсветкой**;
- Подсветка включается **только на время съемки** (экономия энергии и продление срока службы светодиодов);
- Камера **закреплена неподвижно**, однако возможны:
  - Небольшие **вибрации**;
  - Микро-смещения от внешних воздействий.

<img src="readme_images/gas_camera/gas_camera_esp.png" width="50%">
<img src="readme_images/gas_camera/gas_camera_my.png" width="50%">

### 💡 Особенности освещения
- Так как подсветка не горит постоянно, камере требуется **время на адаптацию** при каждом включении;
- Освещение и фон могут **незначительно меняться** от кадра к кадру;
- Это учтено в системе: изображения **выравниваются по яркости** и проходят **бинаризацию**.

### 🧠 Подход к обучению
- Используется **минимальная аугментация** (легкий сдвиг, поворот до 10°);
- Это покрывает возможные вибрации и микро-смещения камеры;
- После предварительной обработки фон и освещение **унифицируются**, чтобы модель «видела» только цифры.

## Сбор данных со счетчика

Счетчик закреплен неподвижно, но возможны вибрации и небольшое смещение камеры из-за внешних воздействий.
<!-- 
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="readme_images/raw_imgs/1.png" width="25%">
  <img src="readme_images/raw_imgs/2.png" width="25%">
  <img src="readme_images/raw_imgs/3.png" width="25%">
  <img src="readme_images/raw_imgs/4.png" width="25%">
  <img src="readme_images/raw_imgs/5.png" width="25%">
  <img src="readme_images/raw_imgs/6.png" width="25%">
  <img src="readme_images/raw_imgs/7.png" width="25%">
</div> -->

| | | |
|:-:|:-:|:-:|
|![](readme_images/raw_imgs/1.png)|![](readme_images/raw_imgs/2.png)|![](readme_images/raw_imgs/3.png)|
|![](readme_images/raw_imgs/4.png)|![](readme_images/raw_imgs/5.png)|![](readme_images/raw_imgs/6.png)|
|![](readme_images/raw_imgs/7.png)|||




С помощью OpenCV выполнена сегментация данных, на основе которых впоследствии были созданы тренировочный и валидационный наборы.

<img src="readme_images/train_dataset.png" width="30%">
<img src="readme_images/val_dataset.png" width="30%">

Так выглядят тренировочные данные после аугментации.

<img src="readme_images/train_dataset_after_augmnt.png" width="50%">

<img src="readme_images/train_val_graphs.png" width="50%">

## Train Classification Report

| Класс | Точность (Precision) | Полнота (Recall) | F1-мера | Поддержка (Support) |
|-------|-------------------|------------------|---------|---------------------|
| 0     | 0.968             | 0.886            | 0.925   | 8204                |
| 1     | 0.953             | 0.974            | 0.964   | 8194                |
| 2     | 0.936             | 0.995            | 0.964   | 8198                |
| 3     | 0.985             | 0.949            | 0.966   | 8203                |
| 4     | 0.996             | 0.971            | 0.984   | 8190                |
| 5     | 0.937             | 0.960            | 0.948   | 8209                |
| 6     | 0.929             | 0.971            | 0.949   | 8204                |
| 7     | 1.000             | 0.940            | 0.969   | 8196                |
| 8     | 0.885             | 0.912            | 0.898   | 8126                |
| 9     | 0.882             | 0.901            | 0.891   | 8196                |

| Метрика | Значение |
|---------|----------|
| **Accuracy (Точность)** | **0.946** |
| **Macro Avg** | 0.947 / 0.946 / 0.946 |
| **Weighted Avg** | 0.947 / 0.946 / 0.946 |
| **Всего образцов** | 81920 |

---

## Train Confusion Matrix

<img src="readme_images/confusion_matrix.png">

## Анализ ошибок модели на валидационном датасете

| Метрика | Значение |
| :--- | :--- |
| 📌 Всего обработано изображений | **2078** |
| ❌ Общее количество ошибок | **7** |
| ✅ Общее количество правильных ответов | **2071** |
| 🎯 Общая точность | **99.66 %** |

<br>

### 📈 Validation Статистика по классам

| Класс | Всего изображений | Ошибок | Точность |
| :---: | :---: | :---: | :---: |
| **0** | 208 | 3 | 98.56 % |
| **1** | 208 | 0 | **100.00 %** |
| **2** | 208 | 0 | **100.00 %** |
| **3** | 208 | 0 | **100.00 %** |
| **4** | 208 | 0 | **100.00 %** |
| **5** | 208 | 1 | 99.52 % |
| **6** | 208 | 0 | **100.00 %** |
| **7** | 208 | 0 | **100.00 %** |
| **8** | 206 | 3 | 98.54 % |
| **9** | 208 | 0 | **100.00 %** |

<img src="readme_images/errors_analyze/analyze_error_0.png" width="50%">
<img src="readme_images/errors_analyze/analyze_error_8.png" width="50%">
<img src="readme_images/errors_analyze/analyze_error_5.png">

---

**Анализ ошибок и предложения по улучшению:**

1. **Проблема толщины линий (0 и 8):** Глядя на ошибочные изображения, можно заметить, что модель склонна путать **0** и **8** в случае толстых или плотных линий. Из-за схожести формы модель чаще отдает предпочтение нулю и ошибается.  
   *Решение:* Добавить в датасет больше примеров с различной толщиной штриха (как толстых, так и тонких цифр) и проследить, как это повлияет на точность.

2. **Плохое распознавание тонких линий (цифра 8):** Модель не всегда находит тонкую перемычку (среднюю палочку) у восьмерки.  
   *Решение:* Расширить обучающую выборку аналогичными примерами, чтобы модель акцентировала внимание на этой ключевой детали.

3. **Проблема с цифрой 5 (чрезмерная аугментация):** В последнем случае с пятеркой часть цифры была сильно обрезана или смещена, вероятно, из-за чрезмерной аугментации (изменений формы, масштаба или положения). В реальных условиях такой сильной деформации не возникает.  
   *Решение:* Уменьшить степень аугментации для валидационных данных, чтобы избежать появления нереалистичных артефактов, которые не должны встречаться на практике.
<!-- 


## 🚀 Метрики производительности

### Время выполнения (на Intel i7, без GPU):

| Этап | Время |
|------|-------|
| Предобработка 1 изображения | 150ms |
| Нарезка 8 цифр | 45ms |
| Инференс модели (1 цифра) | 12ms |
| Инференс всего счетчика (8 цифр) | ~96ms |
| **Полный цикл** | **~291ms** |

### Ресурсы:
- **RAM:** ~2.5GB (в пике)
- **VRAM:** ~1.2GB (если на GPU)
- **Размер модели:** 45MB (сохраненные веса) -->


<!-- 

## 📝 Команды для запуска

```bash
# Установка зависимостей
pip install -r requirements.txt

# Полный пайплайн (предобработка + обучение + тест)
python run_pipeline.py --mode full

# Только предобработка
python src/preprocessing/image_processor.py --input data/raw/ --output data/processed/

# Обучение с логированием в TensorBoard
python src/models/trainer.py --epochs 100 --tensorboard

# Инференс на одном изображении
python src/models/predict.py --image path/to/photo.jpg --visualize

# Оценка на тестовом датасете
python src/utils/metrics.py --test-dir data/test/ --model results/models/best_model.pth

# Генерация отчетов
python src/utils/visualizer.py --generate-report
``` -->


## Технологии

- **Python + PyTorch** — нейросети;
- **OpenCV** — работа с видео и изображениями;
- **Telegram API** — оповещения;
- **YOLO / CNN** — распознавание объектов и цифр.

## Итог

**Полностью автоматизированная система безопасности:**
- Следит за работой газа;
- Отслеживает присутствие человека;
- Предупреждает об опасности.
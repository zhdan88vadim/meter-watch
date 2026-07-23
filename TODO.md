def initialize_redis_defaults():
    """Initialize default values in Redis"""
    defaults = {
        config.REDIS_KEYS['gas_flow']: '0',
        config.REDIS_KEYS['human_last_seen']: str(time.time() - 3600),  # 1 hour ago
        config.REDIS_KEYS['alert_triggered']: '0',
    }
    
    for key, value in defaults.items():
        if not RedisManager.key_exists(key):
            RedisManager.set_key(key, value)
            print(f"✅ Initialized Redis key: {key} = {value}")

# Call this when your app starts
initialize_redis_defaults()





{
  "error": "[Errno 2] No such file or directory: '/app/output/wrong_predictions'"
}




{
  "error": "[Errno 2] No such file or directory: '/app/output/validation'"
}







Дополнительные методы - hget, hdel, rpush, lpop, lrange, smembers, sismember, pipeline



NFO - 📹 Recording STARTED
2026-07-15 16:26:07,964 - app.person_tracker - INFO - 🚶 Person 1 left
2026-07-15 16:26:11,969 - app.video_buffer - INFO - 📹 Stopped recording. Duration: 10.07s
2026-07-15 16:26:11,973 - app.person_tracker - INFO - 🛑 Recording STOPPED







meter-watch/
├── docker-compose.yml
├── .env
│
├── services/                          # Микросервисы
│   ├── person-detector/               # Детектор людей
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── app/
│   │   │   ├── person_tracker.py
│   │   │   ├── telegram_bot.py
│   │   │   └── api.py
│   │   └── run.py
│   │
│   └── cnn-recognizer/                # Распознавание цифр
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── app.py
│       ├── trainer.py
│       ├── models/                    # Архитектуры моделей
│       │   ├── pytorch_model.py
│       │   └── digit_recognizer.py
│       ├── routes/                    # API роуты
│       └── services/                  # Бизнес-логика
│
├── shared/                            # Общая библиотека
│   └── meter_watch_shared/
│       ├── config.py
│       ├── redis_manager.py
│       └── setup.py
│
├── data/                              # ЦЕНТРАЛИЗОВАННЫЕ ДАННЫЕ ⭐
│   ├── raw/                           # Исходные данные с камер
│   │   ├── camera_01/
│   │   │   ├── 2026-07-14/
│   │   │   │   ├── morning_01.jpg
│   │   │   │   └── afternoon_02.jpg
│   │   │   └── 2026-07-15/
│   │   └── camera_02/
│   │
│   ├── processed/                     # После препроцессинга
│   │   ├── crops/                     # Нарезанные цифры
│   │   ├── metadata.json
│   │   └── dataset_info.yaml
│   │
│   ├── train/                         # Тренировочный датасет (70%)
│   │   ├── images/
│   │   └── labels.txt
│   │
│   ├── val/                           # Валидационный (15%)
│   │   ├── images/
│   │   └── labels.txt
│   │
│   └── test/                          # Тестовый (15%, фиксированный!)
│       ├── images/
│       └── labels.txt
│
├── recordings/                        # Видеозаписи с детекциями
│   └── 2026-07-16/
│       └── recording_142440_IDrecording.mp4
│
├── models/                            # Сохраненные веса моделей
│   ├── cnn/
│   │   ├── digit_recognizer.pth
│   │   └── best_model_epoch_99.pth
│   └── hog_svm/
│       ├── hog_svm_model.pkl
│       └── hog_rf_model.pkl
│
├── experiments/                       # Эксперименты и исследования
│   ├── hog_svm/                       # Сравнение с HOG+SVM
│   │   ├── notebooks/
│   │   ├── results/
│   │   └── README.md
│   │
│   └── cnn_experiments/
│       ├── confusion_matrices/
│       ├── misclassifications/
│       └── training_logs/
│
├── notebooks/                         # Jupyter ноутбуки
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_error_analysis.ipynb
│
├── scripts/                           # Вспомогательные скрипты
│   ├── preprocess_data.py
│   ├── split_dataset.py
│   ├── evaluate_model.py
│   └── balance_dataset.py
│
├── docs/                              # Документация
│   ├── README.md
│   ├── RUN.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   └── architecture.md
│
├── tests/                             # Тесты
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── .github/                           # CI/CD
│   └── workflows/
│       ├── test.yml
│       └── deploy.yml
│
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
│
├── Makefile                           # Упрощенные команды
├── pyproject.toml                     # Современный Python
└── .gitignore




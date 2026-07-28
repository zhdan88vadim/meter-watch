39:/media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer$ python 03_train_pretrained_mobi
lenet.py 
📌 Используется устройство: cpu

============================================================
🚀 НАЧАЛО ОБУЧЕНИЯ: mobilenet_v2
============================================================
📊 Классы: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
📊 Train samples: 1040
📊 Val samples: 2078
📊 Количество классов: 10

🔍 Начинаем обучение...
✅ Epoch 1: New best model! Acc: 30.46%
Epoch [1/100] - Loss: 6.9889, Val Acc: 30.46%, LR: 1.00e-04
✅ Epoch 2: New best model! Acc: 48.70%
✅ Epoch 3: New best model! Acc: 61.12%
✅ Epoch 4: New best model! Acc: 72.18%
✅ Epoch 5: New best model! Acc: 76.13%
Epoch [5/100] - Loss: 3.0212, Val Acc: 76.13%, LR: 9.94e-05
✅ Epoch 7: New best model! Acc: 81.04%
✅ Epoch 8: New best model! Acc: 84.74%
✅ Epoch 9: New best model! Acc: 89.41%
Epoch [10/100] - Loss: 1.6348, Val Acc: 89.17%, LR: 9.76e-05
✅ Epoch 12: New best model! Acc: 89.85%
✅ Epoch 13: New best model! Acc: 92.35%
✅ Epoch 15: New best model! Acc: 92.97%
Epoch [15/100] - Loss: 0.9857, Val Acc: 92.97%, LR: 9.46e-05
✅ Epoch 19: New best model! Acc: 93.74%
Epoch [20/100] - Loss: 0.7568, Val Acc: 93.26%, LR: 9.05e-05
✅ Epoch 21: New best model! Acc: 94.18%
✅ Epoch 23: New best model! Acc: 95.77%
Epoch [25/100] - Loss: 0.6673, Val Acc: 95.43%, LR: 8.55e-05
✅ Epoch 29: New best model! Acc: 96.15%
Epoch [30/100] - Loss: 0.5710, Val Acc: 94.56%, LR: 7.96e-05
✅ Epoch 33: New best model! Acc: 97.11%
Epoch [35/100] - Loss: 0.4047, Val Acc: 96.05%, LR: 7.30e-05
✅ Epoch 39: New best model! Acc: 97.26%
Epoch [40/100] - Loss: 0.4220, Val Acc: 94.90%, LR: 6.58e-05
✅ Epoch 43: New best model! Acc: 97.69%
✅ Epoch 45: New best model! Acc: 97.79%
Epoch [45/100] - Loss: 0.3046, Val Acc: 97.79%, LR: 5.82e-05
✅ Epoch 46: New best model! Acc: 97.88%
Epoch [50/100] - Loss: 0.2758, Val Acc: 97.21%, LR: 5.05e-05
Epoch [55/100] - Loss: 0.2290, Val Acc: 97.45%, LR: 4.28e-05
Epoch [60/100] - Loss: 0.2935, Val Acc: 97.02%, LR: 3.52e-05
✅ Epoch 63: New best model! Acc: 98.36%
✅ Epoch 65: New best model! Acc: 98.46%
Epoch [65/100] - Loss: 0.2953, Val Acc: 98.46%, LR: 2.80e-05
✅ Epoch 70: New best model! Acc: 98.51%
Epoch [70/100] - Loss: 0.2901, Val Acc: 98.51%, LR: 2.14e-05
✅ Epoch 73: New best model! Acc: 98.60%
✅ Epoch 74: New best model! Acc: 98.65%
Epoch [75/100] - Loss: 0.2170, Val Acc: 98.60%, LR: 1.55e-05
✅ Epoch 79: New best model! Acc: 98.70%
✅ Epoch 80: New best model! Acc: 98.75%
Epoch [80/100] - Loss: 0.2391, Val Acc: 98.75%, LR: 1.05e-05
Epoch [85/100] - Loss: 0.1901, Val Acc: 98.41%, LR: 6.40e-06
Epoch [90/100] - Loss: 0.2115, Val Acc: 97.79%, LR: 3.42e-06
Epoch [95/100] - Loss: 0.2225, Val Acc: 98.51%, LR: 1.61e-06
✅ Epoch 99: New best model! Acc: 98.85%
Epoch [100/100] - Loss: 0.2472, Val Acc: 97.79%, LR: 1.00e-06

✅ Обучение завершено за 784.0 секунд
🏆 Лучшая точность: 98.85%
💾 Модель сохранена: trained_models/digit_recognizer_mobilenet_v2.pth

============================================================
✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО
============================================================
🏆 Лучшая точность: 98.85%
📁 Модель сохранена: trained_models/digit_recognizer_mobilenet_v2.pth

============================================================
🧪 ТЕСТИРОВАНИЕ МОДЕЛИ
============================================================
📊 Тестовых образцов: 2078
📊 Классы: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

🎯 Общая точность: 97.11%
✅ Правильно: 2018
❌ Ошибок: 60

📊 Точность по классам:
  ✅ 0: 97.60%
  ✅ 1: 100.00%
  ✅ 2: 100.00%
  ✅ 3: 97.60%
  ✅ 4: 98.08%
  ✅ 5: 98.56%
  ✅ 6: 99.52%
  ✅ 7: 98.08%
  ⚠️ 8: 88.35%
  ✅ 9: 93.27%


















  39:/media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer$ python 04_analyze_errors_mobilene
t.py 
Current directory: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer
/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/.conda/lib/python3.11/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
BASE_DIR: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer
MANUAL_RECONGIZED_DATA_DIR: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer/output/manual_recongized_data

============================================================
📊 АНАЛИЗ ОШИБОК МОДЕЛИ MOBILENET
============================================================
📌 Загрузка модели MobileNet из: trained_models/digit_recognizer_mobilenet_v2__full.pth
✅ Загружена полная модель
📊 Классы: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
📊 Всего изображений: 2078
📊 Размер входных изображений: 32x32

🔍 Начинаем анализ ошибок...
   Обработано батчей: 10/33
   Обработано батчей: 20/33
   Обработано батчей: 30/33

============================================================
📊 РЕЗУЛЬТАТЫ АНАЛИЗА
============================================================
📌 Всего обработано изображений: 2078
❌ Общее количество ошибок: 287
✅ Общее количество правильных ответов: 1791
🎯 Общая точность: 86.19%
📊 Средняя уверенность: 80.46%

------------------------------------------------------------
СТАТИСТИКА ПО КЛАССАМ:
------------------------------------------------------------
❌ 8:
  - Всего: 206
  - Ошибок: 84
  - Точность: 59.22%
⚠️ 9:
  - Всего: 208
  - Ошибок: 44
  - Точность: 78.85%
⚠️ 6:
  - Всего: 208
  - Ошибок: 41
  - Точность: 80.29%
⚠️ 5:
  - Всего: 208
  - Ошибок: 32
  - Точность: 84.62%
⚠️ 0:
  - Всего: 208
  - Ошибок: 23
  - Точность: 88.94%
✅ 1:
  - Всего: 208
  - Ошибок: 20
  - Точность: 90.38%
✅ 2:
  - Всего: 208
  - Ошибок: 19
  - Точность: 90.87%
✅ 4:
  - Всего: 208
  - Ошибок: 13
  - Точность: 93.75%
✅ 3:
  - Всего: 208
  - Ошибок: 6
  - Точность: 97.12%
✅ 7:
  - Всего: 208
  - Ошибок: 5
  - Точность: 97.60%
============================================================
✅ Confusion matrix saved!
✅ Misclassifications visualization with Top-3 saved!
✅ Error report with Top-3 saved to: logs/misclassifications/error_report_mobilenet_top3_20260728_171134.txt

📊 Classification Report:
              precision    recall  f1-score   support

           0      0.680     0.889     0.771       208
           1      0.954     0.904     0.928       208
           2      0.936     0.909     0.922       208
           3      0.874     0.971     0.920       208
           4      0.965     0.938     0.951       208
           5      0.941     0.846     0.891       208
           6      0.861     0.803     0.831       208
           7      0.910     0.976     0.942       208
           8      0.646     0.592     0.618       206
           9      0.906     0.788     0.843       208

    accuracy                          0.862      2078
   macro avg      0.867     0.862     0.862      2078
weighted avg      0.868     0.862     0.862      2078

✅ Report saved to: logs/reports/classification_report_mobilenet_20260728_171134.txt



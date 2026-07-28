:/media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer$ python 02_pretrain__analyze_err
ors_top.py 
Current directory: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer
/mnt/ntfs/learn_ML/test_classes/Тестовое Python ML,CV/Тестовое_ML/тестовое_ml/.conda/lib/python3.11/site-packages/torch/cuda/__init__.py:180: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 804: forward compatibility was attempted on non supported HW (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
BASE_DIR: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer
MANUAL_RECONGIZED_DATA_DIR: /media/vadim/1TB_SSD/my_github/meter-watch/services/cnn-recognizer/output/manual_recongized_data
📌 Загрузка модели из: trained_models/digit_recognizer_pretrained_resnet18_full.pth
   Загрузка как полной модели...
✅ Загружена полная модель

🔍 Начинаем анализ ошибок...
   Обработано батчей: 10
   Обработано батчей: 20
   Обработано батчей: 30

============================================================
📊 АНАЛИЗ ОШИБОК ПРЕДОБУЧЕННОЙ МОДЕЛИ
============================================================
📌 Всего обработано изображений: 2078
❌ Общее количество ошибок: 1
✅ Общее количество правильных ответов: 2077
🎯 Общая точность: 99.95%

------------------------------------------------------------
СТАТИСТИКА ПО КЛАССАМ:
------------------------------------------------------------
✅ 8:
  - Всего: 206
  - Ошибок: 1
  - Точность: 99.51%
✅ 0:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 1:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 2:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 3:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 4:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 5:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 6:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 7:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
✅ 9:
  - Всего: 208
  - Ошибок: 0
  - Точность: 100.00%
============================================================
✅ Confusion matrix saved!
✅ Misclassifications visualization with Top-3 saved!
✅ Error report with Top-3 saved to: logs/misclassifications/error_report_pretrained_top3_20260728_163659.txt
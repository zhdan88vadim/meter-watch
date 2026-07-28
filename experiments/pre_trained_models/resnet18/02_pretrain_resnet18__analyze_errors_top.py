# analyze_errors_pretrained.py
from utils.augmentation import AdaptivePreprocess, SquarePad
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from configuration import Config
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from datetime import datetime
import torchvision.models as models
import torch.nn as nn

class PretrainedDigitRecognizer(nn.Module):
    """Предобученная модель для распознавания цифр"""
    def __init__(self, num_classes=10, pretrained=True):
        super(PretrainedDigitRecognizer, self).__init__()
        
        # Используем ResNet18 как базовую модель
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Адаптируем первый слой для 1-канальных изображений
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Копируем веса из оригинального conv1
        with torch.no_grad():
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        
        # Заменяем последний слой на наш классификатор
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Извлечение признаков перед финальным классификатором"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        return features
def load_pretrained_model(model_path, num_classes=10, device='cuda'):
    """
    Загружает предобученную модель из сохраненного файла
    """
    print(f"📌 Загрузка модели из: {model_path}")
    
    # Проверяем существование файла
    if not os.path.exists(model_path):
        print(f"❌ Файл не найден: {model_path}")
        return None
    
    try:
        # Пробуем загрузить как полную модель
        if model_path.endswith('_full.pth'):
            print("   Загрузка как полной модели...")
            model = torch.load(model_path, map_location=device, weights_only=False)
            print("✅ Загружена полная модель")
            
            # Проверяем, что модель на правильном устройстве
            if hasattr(model, 'to'):
                model = model.to(device)
            model.eval()
            return model
        
        # Загрузка state_dict
        else:
            print("   Загрузка как state_dict...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Создаем модель
            model = PretrainedDigitRecognizer(num_classes=num_classes, pretrained=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Загружена модель: {checkpoint.get('model_name', 'unknown')}")
                    print(f"   Точность: {checkpoint.get('accuracy', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    print("✅ Загружен state_dict модели")
            else:
                model = checkpoint
                print("✅ Загружена модель")
            
            model = model.to(device)
            model.eval()
            return model
            
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_model_errors(model_path, dataset_path, num_examples=10, num_classes=10):
    """
    Анализирует ошибки предобученной модели и сохраняет примеры с топ-3 предсказаниями.
    """
    # Загружаем модель
    device = Config.DEVICE
    model = load_pretrained_model(model_path, num_classes, device)
    
    # Загружаем данные
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.Pad(padding=4, fill=0),
        transforms.Resize((28, 28)),               
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.5], std=[0.5])  
              
        # transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((224, 224)),  # ResNet требует 224x224
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229])  # Нормализация для ImageNet
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    # Проверяем соответствие количества классов
    if len(classes) != num_classes:
        print(f"⚠️ Предупреждение: количество классов в датасете ({len(classes)}) "
              f"не совпадает с моделью ({num_classes})")
        num_classes = len(classes)
    
    # Собираем ошибки с топ-3 предсказаниями
    misclassifications = {i: {'images': [], 'predicted': [], 'true': [], 'top3': [], 
                              'confidences': []} 
                          for i in range(len(classes))}
    
    # Счетчики для статистики
    total_samples = 0
    total_errors = 0
    class_correct = {i: 0 for i in range(len(classes))}
    class_total = {i: 0 for i in range(len(classes))}
    
    # Для матрицы ошибок
    all_preds = []
    all_labels = []
    
    print("\n🔍 Начинаем анализ ошибок...")
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Получаем топ-3 предсказания
            probabilities = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, min(3, len(classes)), dim=1)
            
            # Основное предсказание
            _, predicted = torch.max(outputs, 1)
            
            # Обновляем статистику по классам
            for i in range(len(classes)):
                class_mask = labels == i
                class_total[i] += class_mask.sum().item()
                class_correct[i] += ((predicted == labels) & class_mask).sum().item()
            
            total_samples += labels.size(0)
            
            # Сохраняем для матрицы ошибок
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            mask = predicted != labels
            if mask.any():
                error_indices = mask.nonzero(as_tuple=True)[0]
                total_errors += len(error_indices)
                
                for idx in error_indices:
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    
                    # Получаем топ-3 для этого изображения
                    top3_labels = top3_indices[idx].cpu().numpy()
                    top3_probs_values = top3_probs[idx].cpu().numpy()
                    top3_info = [(classes[label], prob) for label, prob in zip(top3_labels, top3_probs_values)]
                    
                    if len(misclassifications[true_label]['images']) < num_examples:
                        misclassifications[true_label]['images'].append(images[idx].cpu())
                        misclassifications[true_label]['predicted'].append(pred_label)
                        misclassifications[true_label]['true'].append(true_label)
                        misclassifications[true_label]['top3'].append(top3_info)
                        misclassifications[true_label]['confidences'].append(
                            top3_probs_values[0]  # Уверенность в основном предсказании
                        )
            
            # Прогресс
            if (batch_idx + 1) % 10 == 0:
                print(f"   Обработано батчей: {batch_idx + 1}")
    
    # Вычисляем точность
    accuracy = (1 - total_errors / total_samples) * 100 if total_samples > 0 else 0
    
    # Выводим статистику
    print("\n" + "="*60)
    print("📊 АНАЛИЗ ОШИБОК ПРЕДОБУЧЕННОЙ МОДЕЛИ")
    print("="*60)
    print(f"📌 Всего обработано изображений: {total_samples}")
    print(f"❌ Общее количество ошибок: {total_errors}")
    print(f"✅ Общее количество правильных ответов: {total_samples - total_errors}")
    print(f"🎯 Общая точность: {accuracy:.2f}%")
    print("\n" + "-"*60)
    print("СТАТИСТИКА ПО КЛАССАМ:")
    print("-"*60)
    
    # Сортировка классов по точности
    class_stats = []
    for i in range(len(classes)):
        class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        class_errors = class_total[i] - class_correct[i]
        class_stats.append((classes[i], class_total[i], class_correct[i], class_errors, class_acc))
    
    # Сортировка по точности (возрастание)
    class_stats.sort(key=lambda x: x[4])
    
    for cls_name, total, correct, errors, acc in class_stats:
        status = "✅" if acc > 90 else "⚠️" if acc > 70 else "❌"
        print(f"{status} {cls_name}:")
        print(f"  - Всего: {total}")
        print(f"  - Ошибок: {errors}")
        print(f"  - Точность: {acc:.2f}%")
    
    print("="*60)
    
    # Визуализируем матрицу ошибок
    if all_preds and all_labels:
        plot_confusion_matrix(all_labels, all_preds, classes)
    
    # Визуализируем ошибки с топ-3
    visualize_errors(misclassifications, classes, num_examples, total_errors, accuracy)
    
    # Сохраняем отчет с топ-3
    save_error_report(misclassifications, classes, total_errors, total_samples, accuracy, 
                      class_correct, class_total, model_path)
    
    return misclassifications, total_errors, accuracy

def plot_confusion_matrix(y_true, y_pred, classes):
    """Визуализация матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_xticks(np.arange(len(classes)))
    ax1.set_yticks(np.arange(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.set_yticklabels(classes)
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax1.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    # Normalized
    im2 = ax2.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_xticks(np.arange(len(classes)))
    ax2.set_yticks(np.arange(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_yticklabels(classes)
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, f'{cm_normalized[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_normalized[i, j] > 0.5 else 'black')
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    os.makedirs('logs/confusion_matrices', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/confusion_matrices/confusion_matrix_pretrained_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved!")

def visualize_errors(misclassifications, classes, num_examples=10, total_errors=0, accuracy=0):
    """Визуализация ошибок с топ-3 предсказаниями"""
    n_classes = len(classes)
    n_rows = min(n_classes, 5)  # Максимум 5 строк для читаемости
    n_cols = min(num_examples, 5)  # Максимум 5 колонок
    
    # Определяем, какие классы показывать (те, у которых есть ошибки)
    classes_with_errors = []
    for i in range(n_classes):
        if len(misclassifications[i]['images']) > 0:
            classes_with_errors.append(i)
    
    if not classes_with_errors:
        print("🎉 Нет ошибок для визуализации!")
        return
    
    # Берем первые n_rows классов с ошибками
    display_classes = classes_with_errors[:n_rows]
    n_display = len(display_classes)
    
    fig, axes = plt.subplots(n_display, n_cols, 
                             figsize=(n_cols * 3.5, n_display * 3))
    
    if n_display == 1:
        axes = axes.reshape(1, -1)
    
    for row, class_idx in enumerate(display_classes):
        class_errors = misclassifications[class_idx]
        n_errors = len(class_errors['images'])
        
        for col in range(n_cols):
            ax = axes[row, col]
            
            if col < n_errors:
                img = class_errors['images'][col]
                pred_label = class_errors['predicted'][col]
                true_label = class_errors['true'][col]
                top3 = class_errors['top3'][col]
                confidence = class_errors['confidences'][col] if 'confidences' in class_errors else 0
                
                # Денормализация для отображения
                img_display = img.clone()
                # Используем mean=0.485, std=0.229 для нормализации
                img_display = img_display * 0.229 + 0.485
                img_display = img_display.clamp(0, 1)
                
                ax.imshow(img_display.squeeze(), cmap='gray')
                
                # Формируем заголовок
                title = f'True: {classes[true_label]}\nPred: {classes[pred_label]}'
                title += f'\nConf: {confidence*100:.1f}%'
                for i, (cls, prob) in enumerate(top3, 1):
                    if i <= 3:  # Показываем топ-3
                        title += f'\n{i}: {cls} ({prob*100:.1f}%)'
                
                ax.set_title(title, fontsize=8, color='red')
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.suptitle(f'Misclassifications of Pretrained Model with Top-3 Predictions\n'
                 f'Total errors: {total_errors}, Accuracy: {accuracy:.2f}%', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/misclassifications/misclassifications_pretrained_top3_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Misclassifications visualization with Top-3 saved!")

def save_error_report(misclassifications, classes, total_errors, total_samples, accuracy,
                      class_correct, class_total, model_path):
    """Сохраняет подробный отчет об ошибках с топ-3 предсказаниями"""
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'logs/misclassifications/error_report_pretrained_top3_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЕТ ОБ ОШИБКАХ ПРЕДОБУЧЕННОЙ МОДЕЛИ (С ТОП-3 ПРЕДСКАЗАНИЯМИ)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📅 Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📌 Путь к модели: {model_path}\n")
        f.write(f"📌 Всего обработано изображений: {total_samples}\n")
        f.write(f"❌ Общее количество ошибок: {total_errors}\n")
        f.write(f"✅ Общее количество правильных ответов: {total_samples - total_errors}\n")
        f.write(f"🎯 Общая точность: {accuracy:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("ДЕТАЛЬНАЯ СТАТИСТИКА ПО КЛАССАМ:\n")
        f.write("-"*60 + "\n")
        
        # Сортировка по точности
        class_stats = []
        for i in range(len(classes)):
            class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
            class_errors = class_total[i] - class_correct[i]
            class_stats.append((classes[i], class_total[i], class_correct[i], class_errors, class_acc))
        
        class_stats.sort(key=lambda x: x[4])
        
        for cls_name, total, correct, errors, acc in class_stats:
            status = "✅" if acc > 90 else "⚠️" if acc > 70 else "❌"
            f.write(f"\n{status} Класс {cls_name}:\n")
            f.write(f"  - Всего примеров: {total}\n")
            f.write(f"  - Правильно: {correct}\n")
            f.write(f"  - Ошибок: {errors}\n")
            f.write(f"  - Точность: {acc:.2f}%\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("ПРИМЕРЫ ОШИБОК С ТОП-3 ПРЕДСКАЗАНИЯМИ:\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(classes)):
            class_errors = misclassifications[i]
            n_errors = len(class_errors['images'])
            if n_errors > 0:
                f.write(f"\nКласс {classes[i]} - найдено ошибок: {n_errors}\n")
                for j in range(min(n_errors, 20)):  # Ограничиваем до 20 примеров
                    top3_str = ", ".join([f"{cls} ({prob*100:.1f}%)" for cls, prob in class_errors['top3'][j]])
                    conf = class_errors['confidences'][j] if 'confidences' in class_errors else 0
                    f.write(f"  {j+1}. True: {classes[class_errors['true'][j]]}, "
                           f"Pred: {classes[class_errors['predicted'][j]]}\n")
                    f.write(f"     Confidence: {conf*100:.1f}%\n")
                    f.write(f"     Top-3: {top3_str}\n")
    
    print(f"✅ Error report with Top-3 saved to: {report_path}")

def analyze_specific_classes(model_path, dataset_path, classes_to_analyze=None):
    """
    Анализирует конкретные классы, которые плохо распознаются
    """
    # Загружаем модель
    device = Config.DEVICE
    model = load_pretrained_model(model_path, num_classes=10, device=device)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    if classes_to_analyze is None:
        classes_to_analyze = classes
    
    # Фильтруем данные по нужным классам
    class_indices = [classes.index(cls) for cls in classes_to_analyze if cls in classes]
    
    if not class_indices:
        print("❌ Не найдены указанные классы")
        return
    
    # Собираем статистику
    class_stats = {cls: {'correct': 0, 'total': 0, 'confusions': {}} for cls in classes_to_analyze}
    
    with torch.no_grad():
        for images, labels in loader:
            for i in range(len(labels)):
                label = labels[i].item()
                if label in class_indices:
                    img = images[i:i+1].to(device)
                    output = model(img)
                    _, pred = torch.max(output, 1)
                    pred = pred.item()
                    
                    cls_name = classes[label]
                    class_stats[cls_name]['total'] += 1
                    
                    if pred == label:
                        class_stats[cls_name]['correct'] += 1
                    else:
                        pred_name = classes[pred]
                        if pred_name not in class_stats[cls_name]['confusions']:
                            class_stats[cls_name]['confusions'][pred_name] = 0
                        class_stats[cls_name]['confusions'][pred_name] += 1
    
    # Выводим результаты
    print("\n" + "="*60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ВЫБРАННЫХ КЛАССОВ")
    print("="*60)
    
    for cls_name in classes_to_analyze:
        stats = class_stats[cls_name]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"\n📊 Класс: {cls_name}")
        print(f"   Всего примеров: {stats['total']}")
        print(f"   Правильно: {stats['correct']}")
        print(f"   Точность: {acc:.2f}%")
        
        if stats['confusions']:
            print("   Ошибки:")
            for pred_name, count in sorted(stats['confusions'].items(), key=lambda x: x[1], reverse=True):
                print(f"     - Распознан как {pred_name}: {count} раз")

if __name__ == "__main__":
    # Путь к сохраненной модели
    model_path = "trained_models/digit_recognizer_pretrained_resnet18_full.pth"
    # или если использовали другое имя:
    # model_path = "trained_models/digit_recognizer_pretrained_resnet18.pth"
    
    # Анализируем ошибки
    misclassifications, total_errors, accuracy = analyze_model_errors(
        model_path=model_path,
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
        num_examples=10,
        num_classes=10  # Укажите количество классов в вашей модели
    )
    
    # Дополнительный анализ проблемных классов
    # analyze_specific_classes(
    #     model_path=model_path,
    #     dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
    #     classes_to_analyze=['0', '1', '2']  # Укажите классы для детального анализа
    # )
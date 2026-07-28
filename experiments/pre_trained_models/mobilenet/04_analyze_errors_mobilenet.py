# analyze_errors_mobilenet.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from configuration import Config
from sklearn.metrics import confusion_matrix, classification_report
import os
from datetime import datetime
import torch.nn as nn
import torchvision.models as models
import warnings
warnings.filterwarnings("ignore")

class MobileNetDigitRecognizer(nn.Module):
    """MobileNetV2 с предобученными весами для распознавания цифр"""
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNetDigitRecognizer, self).__init__()
        
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Адаптация для 1 канала
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Заменяем классификатор
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Извлечение признаков перед финальным классификатором"""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features

def load_mobilenet_model(model_path, num_classes=10, device='cuda'):
    """
    Загружает модель MobileNet из сохраненного файла
    """
    print(f"📌 Загрузка модели MobileNet из: {model_path}")
    
    if model_path.endswith('_full.pth'):
        # Полная модель
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("✅ Загружена полная модель")
    else:
        # Создаем модель
        model = MobileNetDigitRecognizer(num_classes=num_classes, pretrained=False)
        
        # Загружаем state_dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
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

def analyze_model_errors(model_path, dataset_path, num_examples=10, num_classes=10, input_size=32):
    """
    Анализирует ошибки модели MobileNet и сохраняет примеры с топ-3 предсказаниями.
    """
    print("\n" + "="*60)
    print("📊 АНАЛИЗ ОШИБОК МОДЕЛИ MOBILENET")
    print("="*60)
    
    # Загружаем модель
    device = Config.DEVICE
    model = load_mobilenet_model(model_path, num_classes, device)
    
    # Трансформации для MobileNet (32x32)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Загружаем данные
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    print(f"📊 Классы: {classes}")
    print(f"📊 Всего изображений: {len(dataset)}")
    print(f"📊 Размер входных изображений: {input_size}x{input_size}")
    
    # Собираем ошибки с топ-3 предсказаниями
    misclassifications = {i: {'images': [], 'predicted': [], 'true': [], 'top3': [], 'confidences': []} 
                          for i in range(len(classes))}
    
    # Счетчики для статистики
    total_samples = 0
    total_errors = 0
    class_correct = {i: 0 for i in range(len(classes))}
    class_total = {i: 0 for i in range(len(classes))}
    
    # Для матрицы ошибок
    all_preds = []
    all_labels = []
    all_confidences = []
    
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
            
            # Сохраняем уверенность
            confidences = probabilities[range(len(labels)), predicted]
            all_confidences.extend(confidences.cpu().numpy())
            
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
                            top3_probs_values[0]
                        )
            
            # Прогресс
            if (batch_idx + 1) % 10 == 0:
                print(f"   Обработано батчей: {batch_idx + 1}/{len(loader)}")
    
    # Вычисляем точность
    accuracy = (1 - total_errors / total_samples) * 100 if total_samples > 0 else 0
    
    # Выводим статистику
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*60)
    print(f"📌 Всего обработано изображений: {total_samples}")
    print(f"❌ Общее количество ошибок: {total_errors}")
    print(f"✅ Общее количество правильных ответов: {total_samples - total_errors}")
    print(f"🎯 Общая точность: {accuracy:.2f}%")
    print(f"📊 Средняя уверенность: {np.mean(all_confidences)*100:.2f}%")
    
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
    
    # Сохраняем classification report
    save_classification_report(all_labels, all_preds, classes)
    
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
    plt.savefig(f'logs/confusion_matrices/confusion_matrix_mobilenet_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved!")

def visualize_errors(misclassifications, classes, num_examples=10, total_errors=0, accuracy=0):
    """Визуализация ошибок с топ-3 предсказаниями"""
    n_classes = len(classes)
    n_rows = min(n_classes, 5)
    n_cols = min(num_examples, 5)
    
    # Определяем классы с ошибками
    classes_with_errors = []
    for i in range(n_classes):
        if len(misclassifications[i]['images']) > 0:
            classes_with_errors.append(i)
    
    if not classes_with_errors:
        print("🎉 Нет ошибок для визуализации!")
        return
    
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
                
                # Денормализация
                img_display = img.clone()
                img_display = img_display * 0.229 + 0.485
                img_display = img_display.clamp(0, 1)
                
                ax.imshow(img_display.squeeze(), cmap='gray')
                
                # Формируем заголовок
                title = f'True: {classes[true_label]}\nPred: {classes[pred_label]}'
                title += f'\nConf: {confidence*100:.1f}%'
                for i, (cls, prob) in enumerate(top3, 1):
                    if i <= 3:
                        title += f'\n{i}: {cls} ({prob*100:.1f}%)'
                
                ax.set_title(title, fontsize=8, color='red')
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.suptitle(f'Misclassifications of MobileNet with Top-3 Predictions\n'
                 f'Total errors: {total_errors}, Accuracy: {accuracy:.2f}%', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/misclassifications/misclassifications_mobilenet_top3_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Misclassifications visualization with Top-3 saved!")

def save_error_report(misclassifications, classes, total_errors, total_samples, accuracy,
                      class_correct, class_total, model_path):
    """Сохраняет подробный отчет об ошибках с топ-3 предсказаниями"""
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'logs/misclassifications/error_report_mobilenet_top3_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЕТ ОБ ОШИБКАХ МОДЕЛИ MOBILENET (С ТОП-3 ПРЕДСКАЗАНИЯМИ)\n")
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
                for j in range(min(n_errors, 20)):
                    top3_str = ", ".join([f"{cls} ({prob*100:.1f}%)" for cls, prob in class_errors['top3'][j]])
                    conf = class_errors['confidences'][j] if 'confidences' in class_errors else 0
                    f.write(f"  {j+1}. True: {classes[class_errors['true'][j]]}, "
                           f"Pred: {classes[class_errors['predicted'][j]]}\n")
                    f.write(f"     Confidence: {conf*100:.1f}%\n")
                    f.write(f"     Top-3: {top3_str}\n")
    
    print(f"✅ Error report with Top-3 saved to: {report_path}")

def save_classification_report(y_true, y_pred, classes):
    """Сохраняет classification report"""
    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    
    os.makedirs('logs/reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'logs/reports/classification_report_mobilenet_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT - MOBILENET\n")
        f.write("="*60 + "\n")
        f.write(report)
    
    print("\n📊 Classification Report:")
    print(report)
    print(f"✅ Report saved to: {report_path}")

def analyze_problematic_classes(model_path, dataset_path, classes_to_analyze=None):
    """
    Детальный анализ проблемных классов
    """
    print("\n" + "="*60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОБЛЕМНЫХ КЛАССОВ")
    print("="*60)
    
    device = Config.DEVICE
    model = load_mobilenet_model(model_path, num_classes=10, device=device)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    if classes_to_analyze is None:
        classes_to_analyze = classes
    
    class_indices = [classes.index(cls) for cls in classes_to_analyze if cls in classes]
    
    if not class_indices:
        print("❌ Не найдены указанные классы")
        return
    
    class_stats = {cls: {'correct': 0, 'total': 0, 'confusions': {}, 'confidences': []} 
                   for cls in classes_to_analyze}
    
    with torch.no_grad():
        for images, labels in loader:
            for i in range(len(labels)):
                label = labels[i].item()
                if label in class_indices:
                    img = images[i:i+1].to(device)
                    output = model(img)
                    probs = torch.softmax(output, dim=1)
                    _, pred = torch.max(output, 1)
                    pred = pred.item()
                    confidence = probs[0][pred].item()
                    
                    cls_name = classes[label]
                    class_stats[cls_name]['total'] += 1
                    class_stats[cls_name]['confidences'].append(confidence)
                    
                    if pred == label:
                        class_stats[cls_name]['correct'] += 1
                    else:
                        pred_name = classes[pred]
                        if pred_name not in class_stats[cls_name]['confusions']:
                            class_stats[cls_name]['confusions'][pred_name] = 0
                        class_stats[cls_name]['confusions'][pred_name] += 1
    
    print("\n📊 Результаты анализа:")
    print("-"*60)
    
    for cls_name in classes_to_analyze:
        stats = class_stats[cls_name]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
        
        print(f"\n📌 Класс: {cls_name}")
        print(f"   Всего примеров: {stats['total']}")
        print(f"   Правильно: {stats['correct']}")
        print(f"   Точность: {acc:.2f}%")
        print(f"   Средняя уверенность: {avg_conf*100:.2f}%")
        
        if stats['confusions']:
            print("   Ошибки:")
            for pred_name, count in sorted(stats['confusions'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"     - Распознан как {pred_name}: {count} раз ({count/stats['total']*100:.1f}%)")

if __name__ == "__main__":
    # Путь к сохраненной модели
    model_path = "trained_models/digit_recognizer_mobilenet_v2__full.pth"  # Замените на ваш путь
    
    # Или если используете state_dict:
    # model_path = "trained_models/digit_recognizer_mobilenet_v2_epoch49_acc98.1.pth"
    
    # Анализируем ошибки
    misclassifications, total_errors, accuracy = analyze_model_errors(
        model_path=model_path,
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
        num_examples=10,
        num_classes=10,
        input_size=32  # Размер для MobileNet
    )
    
    # Детальный анализ проблемных классов
    # analyze_problematic_classes(
    #     model_path=model_path,
    #     dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
    #     classes_to_analyze=['0', '1', '2']  # Укажите классы для анализа
    # )
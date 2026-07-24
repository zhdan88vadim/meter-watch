# analyze_errors.py
from utils.augmentation import AdaptivePreprocess, SquarePad
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.digit_recognizer import DigitRecognizer
from configuration import Config
from sklearn.metrics import confusion_matrix
from PIL import Image
import os
from datetime import datetime

def analyze_model_errors(model_path, dataset_path, num_examples=10):
    """
    Анализирует ошибки модели и сохраняет примеры с топ-3 предсказаниями.
    """
    # Загружаем модель
    device = Config.DEVICE
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Загружаем данные
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # AdaptivePreprocess(), 
        # SquarePad(fill_white=False),       
        transforms.Resize((28, 28)),
        transforms.Pad(padding=4, fill=0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    # Собираем ошибки с топ-3 предсказаниями
    misclassifications = {i: {'images': [], 'predicted': [], 'true': [], 'top3': []} 
                          for i in range(len(classes))}
    
    # Счетчики для статистики
    total_samples = 0
    total_errors = 0
    class_correct = {i: 0 for i in range(len(classes))}
    class_total = {i: 0 for i in range(len(classes))}
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Получаем топ-3 предсказания
            probabilities = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            # Основное предсказание
            _, predicted = torch.max(outputs, 1)
            
            # Обновляем статистику по классам
            for i in range(len(classes)):
                class_mask = labels == i
                class_total[i] += class_mask.sum().item()
                class_correct[i] += ((predicted == labels) & class_mask).sum().item()
            
            total_samples += labels.size(0)
            
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
    
    # Вычисляем точность
    accuracy = (1 - total_errors / total_samples) * 100
    
    # Выводим статистику
    print("\n" + "="*50)
    print("📊 АНАЛИЗ ОШИБОК МОДЕЛИ")
    print("="*50)
    print(f"📌 Всего обработано изображений: {total_samples}")
    print(f"❌ Общее количество ошибок: {total_errors}")
    print(f"✅ Общее количество правильных ответов: {total_samples - total_errors}")
    print(f"🎯 Общая точность: {accuracy:.2f}%")
    print("\n" + "-"*50)
    print("СТАТИСТИКА ПО КЛАССАМ:")
    print("-"*50)
    
    for i in range(len(classes)):
        class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        class_errors = class_total[i] - class_correct[i]
        print(f"{classes[i]}:")
        print(f"  - Всего: {class_total[i]}")
        print(f"  - Ошибок: {class_errors}")
        print(f"  - Точность: {class_acc:.2f}%")
    
    print("="*50)
    
    # Визуализируем ошибки с топ-3
    visualize_errors(misclassifications, classes, num_examples, total_errors, accuracy)
    
    # Сохраняем отчет с топ-3
    save_error_report(misclassifications, classes, total_errors, total_samples, accuracy, 
                      class_correct, class_total)
    
    return misclassifications, total_errors, accuracy

def visualize_errors(misclassifications, classes, num_examples=10, total_errors=0, accuracy=0):
    """Визуализация ошибок с топ-3 предсказаниями"""
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, num_examples, 
                             figsize=(num_examples * 3, n_classes * 2.5))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for class_idx in range(n_classes):
        class_errors = misclassifications[class_idx]
        n_errors = len(class_errors['images'])
        
        for col in range(num_examples):
            ax = axes[class_idx, col]
            
            if col < n_errors:
                img = class_errors['images'][col]
                pred_label = class_errors['predicted'][col]
                true_label = class_errors['true'][col]
                top3 = class_errors['top3'][col]
                
                # Денормализация
                img_display = img.clone()
                if img_display.min() < 0:
                    img_display = (img_display + 1) / 2
                img_display = img_display.clamp(0, 1)
                
                ax.imshow(img_display.squeeze(), cmap='gray')
                
                # Формируем заголовок с топ-3
                title = f'True: {classes[true_label]}\nPred: {classes[pred_label]}'
                for i, (cls, prob) in enumerate(top3, 1):
                    title += f'\n{i}: {cls} ({prob*100:.1f}%)'
                
                ax.set_title(title, fontsize=7, color='red')
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.suptitle(f'Misclassifications with Top-3 Predictions\nTotal errors: {total_errors}, Accuracy: {accuracy:.2f}%', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Сохраняем
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/misclassifications/misclassifications_top3_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Misclassifications visualization with Top-3 saved!")

def save_error_report(misclassifications, classes, total_errors, total_samples, accuracy,
                      class_correct, class_total):
    """Сохраняет подробный отчет об ошибках с топ-3 предсказаниями в текстовый файл"""
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'logs/misclassifications/error_report_top3_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("ОТЧЕТ ОБ ОШИБКАХ МОДЕЛИ (С ТОП-3 ПРЕДСКАЗАНИЯМИ)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📅 Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📌 Всего обработано изображений: {total_samples}\n")
        f.write(f"❌ Общее количество ошибок: {total_errors}\n")
        f.write(f"✅ Общее количество правильных ответов: {total_samples - total_errors}\n")
        f.write(f"🎯 Общая точность: {accuracy:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("ДЕТАЛЬНАЯ СТАТИСТИКА ПО КЛАССАМ:\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(classes)):
            class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
            class_errors = class_total[i] - class_correct[i]
            f.write(f"\nКласс {classes[i]}:\n")
            f.write(f"  - Всего примеров: {class_total[i]}\n")
            f.write(f"  - Правильно: {class_correct[i]}\n")
            f.write(f"  - Ошибок: {class_errors}\n")
            f.write(f"  - Точность: {class_acc:.2f}%\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("ПРИМЕРЫ ОШИБОК С ТОП-3 ПРЕДСКАЗАНИЯМИ:\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(classes)):
            class_errors = misclassifications[i]
            n_errors = len(class_errors['images'])
            f.write(f"\nКласс {classes[i]} - найдено ошибок: {n_errors}\n")
            for j in range(n_errors):
                top3_str = ", ".join([f"{cls} ({prob*100:.1f}%)" for cls, prob in class_errors['top3'][j]])
                f.write(f"  {j+1}. True: {classes[class_errors['true'][j]]}, "
                       f"Pred: {classes[class_errors['predicted'][j]]}\n")
                f.write(f"     Top-3: {top3_str}\n")
    
    print(f"✅ Error report with Top-3 saved to: {report_path}")

if __name__ == "__main__":
    analyze_model_errors(
        model_path="models/digit_recognizer.pth",
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_binary_val",
        num_examples=10
    )
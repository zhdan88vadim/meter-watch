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
    Анализирует ошибки модели и сохраняет примеры.
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
        SquarePad(fill_white=False),       
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = ImageFolder(root=dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    # Собираем ошибки
    misclassifications = {i: {'images': [], 'predicted': [], 'true': []} 
                          for i in range(len(classes))}
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            mask = predicted != labels
            if mask.any():
                error_indices = mask.nonzero(as_tuple=True)[0]
                
                for idx in error_indices:
                    true_label = labels[idx].item()
                    pred_label = predicted[idx].item()
                    
                    if len(misclassifications[true_label]['images']) < num_examples:
                        misclassifications[true_label]['images'].append(images[idx].cpu())
                        misclassifications[true_label]['predicted'].append(pred_label)
                        misclassifications[true_label]['true'].append(true_label)
    
    # Визуализируем
    visualize_errors(misclassifications, classes, num_examples)
    
    return misclassifications

def visualize_errors(misclassifications, classes, num_examples=10):
    """Визуализация ошибок"""
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, num_examples, 
                             figsize=(num_examples * 2, n_classes * 2))
    
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
                
                # Денормализация
                img_display = img.clone()
                if img_display.min() < 0:
                    img_display = (img_display + 1) / 2
                img_display = img_display.clamp(0, 1)
                
                ax.imshow(img_display.squeeze(), cmap='gray')
                ax.set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}', 
                            fontsize=8, color='red')
                ax.axis('off')
            else:
                ax.axis('off')
    
    plt.suptitle(f'Misclassifications ({num_examples} per class)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Сохраняем
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/misclassifications/misclassifications_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Misclassifications visualization saved!")

if __name__ == "__main__":
    analyze_model_errors(
        model_path="models/digit_recognizer.pth",
        # dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val",
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_old",
        num_examples=10
    )
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
    Analyzes model errors and saves examples with top-3 predictions.
    """
    # Load the model
    device = Config.DEVICE
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load the data
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
    
    # Collect errors with top-3 predictions
    misclassifications = {i: {'images': [], 'predicted': [], 'true': [], 'top3': []} 
                          for i in range(len(classes))}
    
    # Counters for statistics
    total_samples = 0
    total_errors = 0
    class_correct = {i: 0 for i in range(len(classes))}
    class_total = {i: 0 for i in range(len(classes))}
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get top-3 predictions
            probabilities = torch.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            # Main prediction
            _, predicted = torch.max(outputs, 1)
            
            # Update class statistics
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
                    
                    # Get top-3 for this image
                    top3_labels = top3_indices[idx].cpu().numpy()
                    top3_probs_values = top3_probs[idx].cpu().numpy()
                    top3_info = [(classes[label], prob) for label, prob in zip(top3_labels, top3_probs_values)]
                    
                    if len(misclassifications[true_label]['images']) < num_examples:
                        misclassifications[true_label]['images'].append(images[idx].cpu())
                        misclassifications[true_label]['predicted'].append(pred_label)
                        misclassifications[true_label]['true'].append(true_label)
                        misclassifications[true_label]['top3'].append(top3_info)
    
    # Calculate accuracy
    accuracy = (1 - total_errors / total_samples) * 100
    
    # Print statistics
    print("\n" + "="*50)
    print("📊 MODEL ERROR ANALYSIS")
    print("="*50)
    print(f"📌 Total images processed: {total_samples}")
    print(f"❌ Total errors: {total_errors}")
    print(f"✅ Total correct predictions: {total_samples - total_errors}")
    print(f"🎯 Overall accuracy: {accuracy:.2f}%")
    print("\n" + "-"*50)
    print("CLASS-WISE STATISTICS:")
    print("-"*50)
    
    for i in range(len(classes)):
        class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
        class_errors = class_total[i] - class_correct[i]
        print(f"{classes[i]}:")
        print(f"  - Total: {class_total[i]}")
        print(f"  - Errors: {class_errors}")
        print(f"  - Accuracy: {class_acc:.2f}%")
    
    print("="*50)
    
    # Visualize errors with top-3
    visualize_errors(misclassifications, classes, num_examples, total_errors, accuracy)
    
    # Save report with top-3
    save_error_report(misclassifications, classes, total_errors, total_samples, accuracy, 
                      class_correct, class_total)
    
    return misclassifications, total_errors, accuracy

def visualize_errors(misclassifications, classes, num_examples=10, total_errors=0, accuracy=0):
    """Visualization of errors with top-3 predictions"""
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
                
                # Denormalize
                img_display = img.clone()
                if img_display.min() < 0:
                    img_display = (img_display + 1) / 2
                img_display = img_display.clamp(0, 1)
                
                ax.imshow(img_display.squeeze(), cmap='gray')
                
                # Create title with top-3
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
    
    # Save
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/misclassifications/misclassifications_top3_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Misclassifications visualization with Top-3 saved!")

def save_error_report(misclassifications, classes, total_errors, total_samples, accuracy,
                      class_correct, class_total):
    """Saves a detailed error report with top-3 predictions to a text file"""
    os.makedirs('logs/misclassifications', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'logs/misclassifications/error_report_top3_{timestamp}.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("MODEL ERROR REPORT (WITH TOP-3 PREDICTIONS)\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📅 Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"📌 Total images processed: {total_samples}\n")
        f.write(f"❌ Total errors: {total_errors}\n")
        f.write(f"✅ Total correct predictions: {total_samples - total_errors}\n")
        f.write(f"🎯 Overall accuracy: {accuracy:.2f}%\n\n")
        
        f.write("-"*60 + "\n")
        f.write("DETAILED CLASS-WISE STATISTICS:\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(classes)):
            class_acc = (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0
            class_errors = class_total[i] - class_correct[i]
            f.write(f"\nClass {classes[i]}:\n")
            f.write(f"  - Total examples: {class_total[i]}\n")
            f.write(f"  - Correct: {class_correct[i]}\n")
            f.write(f"  - Errors: {class_errors}\n")
            f.write(f"  - Accuracy: {class_acc:.2f}%\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("ERROR EXAMPLES WITH TOP-3 PREDICTIONS:\n")
        f.write("-"*60 + "\n")
        
        for i in range(len(classes)):
            class_errors = misclassifications[i]
            n_errors = len(class_errors['images'])
            f.write(f"\nClass {classes[i]} - errors found: {n_errors}\n")
            for j in range(n_errors):
                top3_str = ", ".join([f"{cls} ({prob*100:.1f}%)" for cls, prob in class_errors['top3'][j]])
                f.write(f"  {j+1}. True: {classes[class_errors['true'][j]]}, "
                       f"Pred: {classes[class_errors['predicted'][j]]}\n")
                f.write(f"     Top-3: {top3_str}\n")
    
    print(f"✅ Error report with Top-3 saved to: {report_path}")

if __name__ == "__main__":
    analyze_model_errors(
        model_path="models/digit_recognizer.pth",
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
        num_examples=10
    )
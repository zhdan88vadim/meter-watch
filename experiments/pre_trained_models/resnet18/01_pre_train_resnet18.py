import os
import torch
import torch.nn as nn
import torch.optim as optim
from augm import AdaptiveAugmentationBuilder
from configuration import Config
import time
from datetime import datetime
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models  # Добавляем импорт для предобученных моделей

def load_config(config_path='config/config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class PretrainedDigitRecognizer(nn.Module):
    """Предобученная модель для распознавания цифр"""
    def __init__(self, num_classes=10, pretrained=True):
        super(PretrainedDigitRecognizer, self).__init__()
        
        # Используем ResNet18 как базовую модель
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Замораживаем веса бэкбона (опционально)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False
        
        # Адаптируем первый слой для 1-канальных изображений (MNIST-style)
        # ResNet18 ожидает 3 канала, мы адаптируем для 1 канала
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Копируем веса из оригинального conv1 (усредняем по каналам RGB)
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
        
        # Инициализация новых слоев
        self._init_new_layers()
    
    def _init_new_layers(self):
        """Инициализация новых слоев"""
        for name, module in self.backbone.fc.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Извлечение признаков перед финальным классификатором"""
        # Проходим через все слои до последнего
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
        
        # Возвращаем признаки до финального классификатора
        return features

class PretrainedModelTrainer:
    """Handles model training with pre-trained models on labeled data"""
    
    def __init__(self, model_name='resnet18', pretrained=True):
        self.device = Config.DEVICE
        self.model = None
        self.train_losses = []
        self.val_accuracies = []
        self.model_name = model_name
        self.pretrained = pretrained

    def prepare_data_from_folders(self, dataset_path, val_dataset_path, batch_size=32, num_workers=4):
        try:
            config = load_config()
            aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])

            train_transform = aug_builder.build_train_transform(
                (config['data']['image_size'], config['data']['image_size'])
            )
            val_transform = aug_builder.build_val_transform(
                (config['data']['image_size'], config['data']['image_size'])
            )
            
            # Load dataset
            train_dataset = ImageFolder(root=dataset_path, transform=train_transform)
            val_dataset = ImageFolder(root=val_dataset_path, transform=val_transform)
            
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True, drop_last=True)
            
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None

    def train_from_folder(self, dataset_path, val_dataset_path, epochs, batch_size, learning_rate):
        """Train the pretrained model on data from a folder-based dataset"""
        print("\n🚀 Starting training with pre-trained model...")
        print(f"📌 Model: {self.model_name}")
        print(f"📌 Using pretrained weights: {self.pretrained}")
        
        try:
            train_loader, val_loader = self.prepare_data_from_folders(dataset_path, val_dataset_path, batch_size)
            
            if train_loader is None:
                return {"success": False, "error": "No training data available"}
            
            if hasattr(train_loader.dataset, 'classes'):
                classes = train_loader.dataset.classes
            else:
                unique_labels = set()
                for _, labels in train_loader:
                    unique_labels.update(labels.numpy())
                classes = [str(i) for i in sorted(unique_labels)]
            
            num_classes = len(classes)
            print(f"📊 Number of classes: {num_classes}")
            print(f"📊 Classes: {classes}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join('runs', f'pretrained_training_{self.model_name}_{timestamp}')
            writer = SummaryWriter(log_dir)
            
            # Инициализация модели с предобученными весами
            self.model = PretrainedDigitRecognizer(
                num_classes=num_classes, 
                pretrained=self.pretrained
            ).to(self.device)
            
            # Настройка оптимизатора
            # Используем разные learning rates для бэкбона и нового классификатора
            backbone_params = []
            classifier_params = []
            
            for name, param in self.model.named_parameters():
                if 'backbone.fc' in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
            
            # Если бэкбон заморожен, не включаем его параметры
            if self.pretrained:
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': learning_rate * 0.1},  # Меньшая LR для предобученных слоев
                    {'params': classifier_params, 'lr': learning_rate}  # Большая LR для новых слоев
                ], weight_decay=1e-4)
            else:
                optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
            
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
            
            self.train_losses = []
            self.val_accuracies = []
            
            start_time = time.time()
            global_step = 0
            best_accuracy = 0
            model_path = None

            all_val_preds = []
            all_val_labels = []            
            
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0

                if epoch % 5 == 0:
                    dataiter = iter(train_loader)
                    images, labels = next(dataiter)
                    img_grid = vutils.make_grid(images[:32], nrow=4, normalize=True)
                    writer.add_image(f'Training/Epoch_{epoch}', img_grid, epoch)

                    dataiter_val = iter(val_loader)
                    images_val, labels_val = next(dataiter_val)
                    img_grid_val = vutils.make_grid(images_val[:32], nrow=4, normalize=True)
                    writer.add_image(f'Val/Epoch_{epoch}', img_grid_val, epoch)

                
                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    if i % 10 == 0:
                        writer.add_scalar('Training/Batch Loss', loss.item(), global_step)
                    
                    global_step += 1
                
                avg_loss = running_loss / len(train_loader)
                self.train_losses.append(avg_loss)
                writer.add_scalar('Training/Epoch Loss', avg_loss, epoch)
                
                if val_loader:
                    val_accuracy, val_preds, val_labels = self._validate_with_predictions(val_loader, criterion)
                    self.val_accuracies.append(val_accuracy)
                    scheduler.step()
                    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
                    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
                    
                    all_val_preds.extend(val_preds)
                    all_val_labels.extend(val_labels)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    progress = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}"
                    if val_loader:
                        progress += f", Val Acc: {val_accuracy:.2f}%"
                        progress += f", LR: {scheduler.get_last_lr()[0]:.2e}"
                    print(progress)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    model_path = self._save_model()
                    print(f"✅ Saved new best model with accuracy: {val_accuracy:.2f}%")                    


            # After training, create confusion matrix
            if all_val_preds and all_val_labels:
                self._plot_confusion_matrix(
                    all_val_labels, 
                    all_val_preds, 
                    classes,
                    writer,
                    epoch
                )
                
                report = classification_report(
                    all_val_labels, 
                    all_val_preds, 
                    target_names=classes,
                    digits=3
                )
                print("\n📊 Classification Report:")
                print(report)
                
                report_path = Path("classification_report_pretrained.txt")
                with open(report_path, 'w') as f:
                    f.write(f"Classification Report - {self.model_name}\n")
                    f.write("=" * 50 + "\n")
                    f.write(report)
                    f.write(f"\n\nBest Accuracy: {best_accuracy:.2f}%")
                print(f"✅ Report saved to {report_path}")

            training_time = time.time() - start_time
            print(f"\n✅ Training completed in {training_time:.2f}s")
            
            if val_loader:
                try:
                    self._log_embeddings(writer, val_loader, classes)
                except Exception as e:
                    pass
            
            self.log_detailed_embeddings(writer, train_loader, val_loader, classes, max_samples=500)
            
            writer.flush()
            writer.close()
            
            model_path = self._save_model()
            
            return {
                "success": True,
                "epochs": epochs,
                "final_loss": self.train_losses[-1],
                "final_accuracy": self.val_accuracies[-1] if self.val_accuracies else None,
                "training_time": training_time,
                "model_path": model_path,
                "num_samples": len(train_loader.dataset),
                "dataset_path": dataset_path,
                "tensorboard_dir": log_dir,
                "model_name": self.model_name,
                "pretrained": self.pretrained
            }
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def log_detailed_embeddings(self, writer, train_loader, val_loader, classes, max_samples=500):
        """Log embeddings from model's intermediate layer"""
        
        all_images = []
        all_labels = []
        
        for images, labels in train_loader:
            all_images.append(images)
            all_labels.append(labels)
            if sum(len(l) for l in all_labels) >= max_samples:
                break
        
        images = torch.cat(all_images)[:max_samples]
        labels = torch.cat(all_labels)[:max_samples]
        
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for i in range(0, len(images), 32):
                batch = images[i:i+32].to(self.device)
                batch_features = self.model.get_features(batch)
                features.append(batch_features.cpu())
        
        features = torch.cat(features)
        
        metadata = []
        for i, label in enumerate(labels):
            metadata.append(f"{classes[label]}_{i:03d}")
        
        writer.add_embedding(
            features,
            metadata=metadata,
            label_img=images,
            global_step=0,
            tag='Model_Features_Embeddings'
        )
        
        writer.add_histogram('Features/Distribution', features, 0)
        writer.add_scalar('Features/Mean', features.mean().item(), 0)
        writer.add_scalar('Features/Std', features.std().item(), 0)
        writer.flush()

    def _log_embeddings(self, writer, data_loader, classes, n_samples=100):
        """Simple embedding logging for TensorBoard"""
        if not data_loader:
            return
        
        try:
            dataiter = iter(data_loader)
            images, labels = next(dataiter)
            
            images = images[:n_samples]
            labels = labels[:n_samples]
            
            features = images.view(images.size(0), -1)
            metadata = [classes[label] for label in labels]
            
            writer.add_embedding(
                features,
                metadata=metadata,
                label_img=images,
                global_step=0,
                tag='embeddings'
            )
            writer.flush()
        except Exception as e:
            pass

    def _save_model(self):
        """Save the trained model with metadata"""
        if self.model is None:
            return None

        # Сохраняем модель с метаданными
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'pretrained': self.pretrained,
            'num_classes': self.model.backbone.fc[-1].out_features
        }
        
        model_path = Config.TRAINED_MODEL_PATH.replace('.pth', f'_pretrained_{self.model_name}.pth')
        torch.save(save_dict, model_path)
        
        # Сохраняем полную модель для загрузки
        torch.save(self.model, model_path.replace('.pth', '_full.pth'))
        
        return model_path

    def _plot_confusion_matrix(self, y_true, y_pred, classes, writer, epoch):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    ax=ax1, cbar=True)
        ax1.set_title(f'Confusion Matrix (Counts) - {self.model_name}', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                    xticklabels=classes, yticklabels=classes,
                    ax=ax2, cbar=True)
        ax2.set_title(f'Confusion Matrix (Normalized) - {self.model_name}', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        
        writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
        
        os.makedirs('logs/confusion_matrices', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.savefig(f'logs/confusion_matrices/confusion_matrix_{self.model_name}_{timestamp}.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("\n📊 Class-wise Accuracy:")
        print("=" * 40)
        for i, class_name in enumerate(classes):
            if cm[i].sum() > 0:
                class_acc = cm[i, i] / cm[i].sum() * 100
                print(f"{class_name}: {class_acc:.2f}%")
        
        accuracy = np.trace(cm) / np.sum(cm) * 100
        print(f"\n📊 Overall Accuracy: {accuracy:.2f}%")

    def _validate_with_predictions(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_preds = []
        all_labels = []        

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())                
        
        accuracy = 100 * correct / total
        return accuracy, all_preds, all_labels


if __name__ == '__main__':
    # Выбор модели: 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0' и т.д.
    # Для использования предобученной модели установите pretrained=True
    trainer = PretrainedModelTrainer(
        model_name='resnet18',  # Можно изменить на другую модель
        pretrained=True         # True - использовать предобученные веса, False - обучать с нуля
    )
    
    result = trainer.train_from_folder(
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_train",
        val_dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val",
        epochs=40,
        batch_size=64,
        learning_rate=0.0005
    )
    
    if result['success']:
        print(f"\n✅ Training successful!")
        print(f"📌 Model: {result['model_name']}")
        print(f"📌 Best accuracy: {result['final_accuracy']:.2f}%")
        print(f"📌 Model saved at: {result['model_path']}")
        print(f"📌 TensorBoard logs: {result['tensorboard_dir']}")
    else:
        print(f"\n❌ Training failed: {result['error']}")
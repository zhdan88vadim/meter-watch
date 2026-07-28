# train_pretrained_mobilenet.py
import os
from utils.augmentation import ExtractLetterWithMargin, SquarePadAdaptBackground, AdaptivePreprocess, RemoveSmallObjects
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path='config/config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class MobileNetDigitRecognizer(nn.Module):
    """MobileNetV2 с предобученными весами для распознавания цифр"""
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNetDigitRecognizer, self).__init__()
        
        # Используем MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Адаптация для 1 канала (вход 32x32 или 28x28)
        # MobileNetV2 ожидает 3 канала, меняем на 1
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
        
        # Инициализация новых слоев
        self._init_new_layers()
    
    def _init_new_layers(self):
        """Инициализация весов как в вашей модели DigitRecognizer"""
        for module in self.backbone.classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Извлечение признаков перед финальным классификатором"""
        # Проходим через все слои кроме классификатора
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features

class ModelTrainer:
    def __init__(self, model_name='mobilenet_v2', input_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = model_name
        self.input_size = input_size
        self.train_losses = []
        self.val_accuracies = []
        print(f"📌 Используется устройство: {self.device}")
    
    def prepare_data(self, dataset_path, val_dataset_path, batch_size=32, num_workers=4):
        """Подготовка данных с правильными трансформациями"""
        
        adaptive_preprocess_params = {
            'blur_ksize': 7,           # Уменьшено с 7 до 3
            'blur_sigma': 5,           # Уменьшено с 5 до 1
            'adaptive_block_size': 57, # Уменьшено с 57 до 11 (должно быть > 1 и нечетное)
            'adaptive_c': 5,           # Уменьшено с 5 до 3
            'morph_kernel': 2,         # Уменьшено с 2 до 1
            'morph_iter': 1            # Оставлено 1
        }

        # Трансформации для обучения
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            ExtractLetterWithMargin(margin=20, fill_white=None),
            SquarePadAdaptBackground(min_size=128),
            AdaptivePreprocess(apply_prob=1, params=adaptive_preprocess_params),
            # transforms.Resize((128, 128)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,              # Угол поворота в градусах (-180 до 180) или (min, max)
                translate=(0.1, 0.1),   # Сдвиг: (по_горизонтали_макс%, по_вертикали_макс%)
                scale=(0.7, 1.1),       # Масштабирование: (мин_коэф, макс_коэф)
                shear=4,                # Наклон в градусах или (min, max) или (x_min, x_max, y_min, y_max)
                interpolation=2,        # Метод интерполяции (NEAREST=0, BILINEAR=2, BICUBIC=3)
                fill=0,                 # Цвет заливки для новых пикселей
            ),
            transforms.CenterCrop((90, 90)),
            transforms.Resize((32, 32)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((32, 32)),
            RemoveSmallObjects(min_area=5, apply_prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Трансформации для валидации
        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.Pad(padding=4, fill=0),
            transforms.Resize((32, 32)),            
            # transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        # Загрузка датасетов
        train_dataset = ImageFolder(root=dataset_path, transform=train_transform)
        val_dataset = ImageFolder(root=val_dataset_path, transform=val_transform)
        
        print(f"📊 Классы: {train_dataset.classes}")
        print(f"📊 Train samples: {len(train_dataset)}")
        print(f"📊 Val samples: {len(val_dataset)}")
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers, 
            pin_memory=True
        )
        
        return train_loader, val_loader, train_dataset.classes
    
    def train(self, dataset_path, val_dataset_path, epochs=50, batch_size=64, learning_rate=0.001):
        """Обучение модели"""
        print("\n" + "="*60)
        print(f"🚀 НАЧАЛО ОБУЧЕНИЯ: {self.model_name}")
        print("="*60)
        
        # Подготовка данных
        train_loader, val_loader, classes = self.prepare_data(
            dataset_path, val_dataset_path, batch_size
        )
        
        num_classes = len(classes)
        print(f"📊 Количество классов: {num_classes}")
        
        # Создание модели
        self.model = MobileNetDigitRecognizer(
            num_classes=num_classes, 
            pretrained=True
        ).to(self.device)
        
        # Настройка оптимизатора
        # Разные learning rates для разных частей модели
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate * 0.1},
            {'params': classifier_params, 'lr': learning_rate}
        ], weight_decay=1e-4)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join('runs', f'mobilenet_training_{timestamp}')
        writer = SummaryWriter(log_dir)
        
        # Переменные для отслеживания
        self.train_losses = []
        self.val_accuracies = []
        best_accuracy = 0
        best_model_path = None
        
        print("\n🔍 Начинаем обучение...")
        start_time = datetime.now()
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Логирование
                if batch_idx % 50 == 0:
                    step = epoch * len(train_loader) + batch_idx
                    writer.add_scalar('Training/Batch_Loss', loss.item(), step)
            
            avg_loss = running_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
            
            # Validation
            val_accuracy, val_loss = self.validate(val_loader, criterion)
            self.val_accuracies.append(val_accuracy)
            scheduler.step()
            
            writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
            writer.add_scalar('Validation/Loss', val_loss, epoch)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
            
            # Сохранение лучшей модели
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = self.save_model(val_accuracy)
                print(f"✅ Epoch {epoch+1}: New best model! Acc: {val_accuracy:.2f}%")
            
            # Прогресс
            if (epoch + 1) % 5 == 0 or epoch == 0:
                progress = (f"Epoch [{epoch+1}/{epochs}] - "
                           f"Loss: {avg_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.2f}%, "
                           f"LR: {scheduler.get_last_lr()[0]:.2e}")
                print(progress)
        
        # Завершение обучения
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Обучение завершено за {training_time:.1f} секунд")
        print(f"🏆 Лучшая точность: {best_accuracy:.2f}%")
        print(f"💾 Модель сохранена: {best_model_path}")
        
        writer.close()
        
        # Финальная валидация
        final_accuracy, _, _, _ = self.validate_full(val_loader)
        
        return {
            'success': True,
            'best_accuracy': best_accuracy,
            'final_accuracy': final_accuracy,
            'model_path': best_model_path,
            'training_time': training_time,
            'num_classes': num_classes,
            'classes': classes
        }
    
    def validate(self, val_loader, criterion):
        """Валидация модели"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = val_loss / len(val_loader)
        return accuracy, avg_loss
    
    def validate_full(self, val_loader):
        """Полная валидация с возвратом предсказаний"""
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        return accuracy, all_preds, all_labels, total
    
    def save_model(self, accuracy):
        """Сохранение модели"""
        os.makedirs('trained_models', exist_ok=True)
        
        model_path = f'trained_models/digit_recognizer_{self.model_name}.pth'
        
        # Сохраняем полную модель для легкой загрузки
        torch.save(self.model, model_path.replace('.pth', '_full.pth'))
        
        # Сохраняем state_dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'model_name': self.model_name,
            'input_size': self.input_size
        }, model_path)
        
        return model_path

def plot_training_history(trainer):
    """Визуализация истории обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Потери
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Точность
    ax2.plot(trainer.val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    os.makedirs('logs/training', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/training/training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_model(model_path, test_dataset_path, input_size=32):
    """Тестирование сохраненной модели"""
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загрузка модели
    if model_path.endswith('_full.pth'):
        model = torch.load(model_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        num_classes = checkpoint.get('num_classes', 10)
        model = MobileNetDigitRecognizer(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Загрузка данных
    dataset = ImageFolder(root=test_dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    classes = dataset.classes
    
    print(f"📊 Тестовых образцов: {len(dataset)}")
    print(f"📊 Классы: {classes}")
    
    # Тестирование
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"\n🎯 Общая точность: {accuracy:.2f}%")
    print(f"✅ Правильно: {correct}")
    print(f"❌ Ошибок: {total - correct}")
    
    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    os.makedirs('logs/test', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'logs/test/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Классовая точность
    print("\n📊 Точность по классам:")
    for i, class_name in enumerate(classes):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum() * 100
            status = "✅" if class_acc > 90 else "⚠️" if class_acc > 70 else "❌"
            print(f"  {status} {class_name}: {class_acc:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # Пути к данным
    train_path = "/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_train/"
    val_path = "/media/vadim/1TB_SSD/my_github/meter-watch/data/dataset_binary_val/"
    
    # Создание тренера
    trainer = ModelTrainer(
        model_name='mobilenet_v2',
        input_size=32  # MobileNetV2 работает с 32x32
    )
    
    # Обучение
    result = trainer.train(
        dataset_path=train_path,
        val_dataset_path=val_path,
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    if result['success']:
        print("\n" + "="*60)
        print("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
        print("="*60)
        print(f"🏆 Лучшая точность: {result['best_accuracy']:.2f}%")
        print(f"📁 Модель сохранена: {result['model_path']}")
        
        # Визуализация
        plot_training_history(trainer)
        
        # Тестирование на валидационном наборе
        test_model(
            model_path=result['model_path'],
            test_dataset_path=val_path,
            input_size=32
        )
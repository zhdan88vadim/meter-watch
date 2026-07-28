Да, есть несколько предобученных моделей, которые работают с меньшим разрешением:

## 1. **MobileNet** (самый легкий, работает с 32x32+)
```python
import torchvision.models as models
import torch.nn as nn

class MobileNetDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(MobileNetDigitRecognizer, self).__init__()
        
        # MobileNetV2 - работает с 32x32 и выше
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Адаптация для 1 канала
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Замена классификатора
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

## 2. **EfficientNet-Lite0** (работает с 32x32+)
```python
class EfficientNetDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetDigitRecognizer, self).__init__()
        
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Адаптация для 1 канала
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Замена классификатора
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

## 3. **SqueezeNet** (очень маленький, работает с 32x32+)
```python
class SqueezeNetDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(SqueezeNetDigitRecognizer, self).__init__()
        
        self.backbone = models.squeezenet1_1(pretrained=pretrained)
        
        # Адаптация для 1 канала
        self.backbone.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        
        # Замена классификатора
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        return self.backbone(x).squeeze()
```

## 4. **ShuffleNetV2** (легкий, работает с 32x32+)
```python
class ShuffleNetDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ShuffleNetDigitRecognizer, self).__init__()
        
        self.backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
        
        # Адаптация для 1 канала
        self.backbone.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Замена классификатора
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
```

## 5. **Лучший вариант для MNIST - обучение с нуля или дообучение легкой модели**

Для MNIST (28x28) лучше всего использовать легкие модели:

```python
# Простая, но эффективная модель для MNIST
class LightweightDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10):
        super(LightweightDigitRecognizer, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

## Сравнение моделей:

| Модель | Min размер | Параметры | Скорость | Точность на MNIST |
|--------|------------|-----------|----------|-------------------|
| ResNet18 | 224x224 | 11.7M | Медленная | ~99.5% |
| MobileNetV2 | 32x32 | 3.5M | Быстрая | ~99.4% |
| EfficientNet-B0 | 32x32 | 5.3M | Средняя | ~99.5% |
| SqueezeNet | 32x32 | 1.2M | Очень быстрая | ~99.0% |
| ShuffleNetV2 | 32x32 | 2.3M | Быстрая | ~99.2% |

## Рекомендация для вашего случая:

```python
# Используйте MobileNetV2 с входом 32x32
class BestDigitRecognizer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(BestDigitRecognizer, self).__init__()
        
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Адаптация для 1 канала и маленького размера
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Убираем первый stride чтобы не уменьшать слишком сильно
        # или используем 28x28 с паддингом до 32x32
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Трансформации для 32x32
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),  # или (28, 28) с паддингом
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
```

**Для вашего случая рекомендую MobileNetV2** - она:
- Работает с 32x32 (можно даже 28x28 с небольшими изменениями)
- Быстрая
- Маленькая (3.5M параметров)
- Хорошо работает с предобученными весами

Хотите, я перепишу ваш код обучения с использованием MobileNetV2?
import torch
import torch.nn as nn

class DigitRecognizer(nn.Module):
    """PyTorch model for digit recognition with Batch Normalization"""
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # Инициализация весов (Kaiming для ReLU)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Change of form
        x = x.view(-1, 64 * 7 * 7)
        
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """
        Extract features before the final classification layer
        Returns features from the layer before fc2 (the 128-dim vector)
        """

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        x = x.view(-1, 64 * 7 * 7)
        
        # Features before final classification (with BatchNorm, but without Dropout)
        x = self.relu(self.bn3(self.fc1(x)))
        
        return x  # 128-dimensional feature vector
    
    def get_intermediate_features(self, x):
        """Get features at different levels (for debugging)"""
        features = {}
        
        # After the first convolutional block
        x1 = self.pool(self.relu(self.bn1(self.conv1(x))))
        features['after_conv1'] = x1
        features['after_conv1_stats'] = {
            'mean': x1.mean().item(),
            'std': x1.std().item(),
            'min': x1.min().item(),
            'max': x1.max().item()
        }
        
        # After the second convolutional block
        x2 = self.pool(self.relu(self.bn2(self.conv2(x1))))
        features['after_conv2'] = x2
        features['after_conv2_stats'] = {
            'mean': x2.mean().item(),
            'std': x2.std().item(),
            'min': x2.min().item(),
            'max': x2.max().item()
        }
        
        # After changing the shape
        x3 = x2.view(-1, 64 * 7 * 7)
        features['flattened'] = x3
        
        # After the first fully connected layer (features)
        x4 = self.relu(self.bn3(self.fc1(x3)))
        features['after_fc1'] = x4
        features['after_fc1_stats'] = {
            'mean': x4.mean().item(),
            'std': x4.std().item(),
            'min': x4.min().item(),
            'max': x4.max().item()
        }
        
        return features
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.augmentation import AdaptiveAugmentationBuilder
from configuration import Config
import time
from datetime import datetime
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from models.digit_recognizer import DigitRecognizer
import yaml

def load_config(config_path='config/config.yaml'):
    """Loads the configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelTrainer:
    """Handles model training on labeled data"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = None
        self.train_losses = []
        self.val_accuracies = []

    def prepare_data_from_folders(self, dataset_path, batch_size=32, validation_split=0.2, num_workers=4):
        try:
            from torchvision import transforms
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader, random_split, Subset

            config = load_config()
            aug_builder = AdaptiveAugmentationBuilder(base_size=config['data']['image_size'])
                
            # # Define transforms
            # train_transform = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),
            #     transforms.Resize((28, 28)),
            #     transforms.RandomRotation(degrees=10),
            #     transforms.RandomAffine(
            #         degrees=0,
            #         translate=(0.2, 0.2),
            #         scale=(0.9, 1.1),
            #         shear=5
            #     ),
            #     transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            #     transforms.ToTensor(),
            #     transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
            #     transforms.Normalize((0.5,), (0.5,))
            # ])
            
            # val_transform = transforms.Compose([
            #     transforms.Grayscale(num_output_channels=1),
            #     transforms.Resize((28, 28)),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5,), (0.5,))
            # ])

            train_transform = aug_builder.build_train_transform(
                (config['data']['image_size'], config['data']['image_size'])
            )
            val_transform = aug_builder.build_val_transform(
                (config['data']['image_size'], config['data']['image_size'])
            )
            
            # Load dataset
            full_dataset = ImageFolder(root=dataset_path)
            
            # Split indices
            train_size = int((1 - validation_split) * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_indices, val_indices = random_split(
                range(len(full_dataset)), 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Get index lists
            train_idx = train_indices.indices
            val_idx = val_indices.indices
            
            # Create datasets with appropriate transforms
            train_dataset = ImageFolder(root=dataset_path, transform=train_transform)
            val_dataset = ImageFolder(root=dataset_path, transform=val_transform)
            
            # Apply indices
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(val_dataset, val_idx)
            
            # DataLoaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
            
            return train_loader, val_loader, full_dataset
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def train_from_folder(self, dataset_path, epochs, batch_size, learning_rate):
        """Train the model on data from a folder-based dataset with TensorBoard logging"""
        print("\nStarting model training...")
        
        try:
            train_loader, val_loader, dataset = self.prepare_data_from_folders(dataset_path, batch_size)
            
            if train_loader is None:
                return {"success": False, "error": "No training data available"}
            
            if hasattr(dataset, 'classes'):
                classes = dataset.classes
            elif hasattr(train_loader.dataset.dataset, 'classes'):
                classes = train_loader.dataset.dataset.classes
            else:
                unique_labels = set()
                for _, labels in train_loader:
                    unique_labels.update(labels.numpy())
                classes = [str(i) for i in sorted(unique_labels)]
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join('runs', f'folder_training_{timestamp}')
            writer = SummaryWriter(log_dir)
            
            dataiter = iter(train_loader)
            images, labels = next(dataiter)
            
            self.model = DigitRecognizer().to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
            
            self.train_losses = []
            self.val_accuracies = []
            
            start_time = time.time()
            global_step = 0
            best_accuracy = 0
            model_path = None
            
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0

                if epoch % 5 == 0:  # Every 10 epochs
                    dataiter = iter(train_loader)
                    images, labels = next(dataiter)
                    img_grid = vutils.make_grid(images[:32], nrow=4, normalize=True)
                    writer.add_image(f'Training/Epoch_{epoch}', img_grid, epoch)
                
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
                    val_accuracy = self._validate(val_loader, criterion)
                    self.val_accuracies.append(val_accuracy)
                    scheduler.step(100 - val_accuracy)
                    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
                    writer.add_scalar('Validation/Loss', 100 - val_accuracy, epoch)
                
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    progress = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}"
                    if val_loader:
                        progress += f", Val Acc: {val_accuracy:.2f}%"
                    print(progress)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    model_path = self._save_model()
                    print(f"Saved new best model with accuracy: {val_accuracy:.2f}%")                    
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f}s")
            
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
                "tensorboard_dir": log_dir
            }
            
        except Exception as e:
            print(f"Training error: {e}")
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
        """Save the trained model with optional prefix"""
        if self.model is None:
            return None

        torch.save(self.model.state_dict(), Config.MODEL_PATH)
        
        return Config.MODEL_PATH

    def _validate(self, val_loader, criterion):
        """Validate the model"""
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
        return accuracy


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_from_folder(
        dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset",
        # dataset_path=Config.TRAINING_DATA_DIR,
        epochs=50,
        batch_size=64,
        learning_rate=0.0005
    )
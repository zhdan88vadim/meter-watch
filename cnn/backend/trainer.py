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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

    def prepare_data_from_folders(self, dataset_path, val_dataset_path, batch_size=32, validation_split=0.2, num_workers=4):
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
            
            # shuffle=True --- for debug image only
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True, drop_last=True)
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def train_from_folder(self, dataset_path, val_dataset_path, epochs, batch_size, learning_rate):
        """Train the model on data from a folder-based dataset with TensorBoard logging"""
        print("\nStarting model training...")
        
        try:
            train_loader, val_loader = self.prepare_data_from_folders(dataset_path, val_dataset_path, batch_size)
            
            if train_loader is None:
                return {"success": False, "error": "No training data available"}
            
            if hasattr(train_loader.dataset, 'classes'):
                classes = train_loader.dataset.classes
            else:
                # Fallback: extract unique labels from the dataset
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

            # Store predictions for final confusion matrix
            all_val_preds = []
            all_val_labels = []            
            
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0

                if epoch % 5 == 0:  # Every 10 epochs
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
                    scheduler.step(100 - val_accuracy)
                    writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
                    writer.add_scalar('Validation/Loss', 100 - val_accuracy, epoch)
                    
                    # Store predictions for confusion matrix
                    all_val_preds.extend(val_preds)
                    all_val_labels.extend(val_labels)


                if (epoch + 1) % 5 == 0 or epoch == 0:
                    progress = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}"
                    if val_loader:
                        progress += f", Val Acc: {val_accuracy:.2f}%"
                    print(progress)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    model_path = self._save_model()
                    print(f"Saved new best model with accuracy: {val_accuracy:.2f}%")                    


            # After training, create confusion matrix
            if all_val_preds and all_val_labels:
                self._plot_confusion_matrix(
                    all_val_labels, 
                    all_val_preds, 
                    classes,
                    writer,
                    epoch
                )
                
                # Generate classification report
                report = classification_report(
                    all_val_labels, 
                    all_val_preds, 
                    target_names=classes,
                    digits=3
                )
                print("\n📊 Classification Report:")
                print(report)
                
                # Save report to file
                report_path = Path("classification_report.txt")
                with open(report_path, 'w') as f:
                    f.write("Classification Report\n")
                    f.write("=" * 50 + "\n")
                    f.write(report)
                    f.write(f"\n\nBest Accuracy: {best_accuracy:.2f}%")
                print(f"✅ Report saved to {report_path}")

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

    def log_detailed_embeddings(self, writer, train_loader, classes, max_samples=500):
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

    def _plot_confusion_matrix(self, y_true, y_pred, classes, writer, epoch):
            """
            Создает и сохраняет confusion matrix в TensorBoard и локально
            """
            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Normalize confusion matrix (percentage)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot 1: Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=classes, yticklabels=classes,
                        ax=ax1, cbar=True)
            ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
            ax1.set_xlabel('Predicted', fontsize=12)
            ax1.set_ylabel('True', fontsize=12)
            
            # Plot 2: Normalized (percentages)
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
                        xticklabels=classes, yticklabels=classes,
                        ax=ax2, cbar=True)
            ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
            ax2.set_xlabel('Predicted', fontsize=12)
            ax2.set_ylabel('True', fontsize=12)
            
            plt.tight_layout()
            
            # Save to TensorBoard
            writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
            
            # Save locally
            os.makedirs('logs/confusion_matrices', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig.savefig(f'logs/confusion_matrices/confusion_matrix_{timestamp}.png', 
                        dpi=300, bbox_inches='tight', facecolor='white')
            fig.savefig(f'logs/confusion_matrices/confusion_matrix_epoch_{epoch}.png', 
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # Print class-wise accuracy
            print("\n📊 Class-wise Accuracy:")
            print("=" * 40)
            for i, class_name in enumerate(classes):
                if cm[i].sum() > 0:
                    class_acc = cm[i, i] / cm[i].sum() * 100
                    print(f"{class_name}: {class_acc:.2f}%")
            
            # Calculate overall metrics
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
                
                # Store predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())                
        
        accuracy = 100 * correct / total
        return accuracy, all_preds, all_labels


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_from_folder(
        # dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset",
        # dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_test",
        # dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val",
        dataset_path = "/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_clean/",
        # val_dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val/",
        # val_dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_low_transform/",
        val_dataset_path="/media/vadim/1TB_SSD/my_github/meter-watch/dataset_val_low_transform_bi/",
        # dataset_path=Config.TRAINING_DATA_DIR,
        epochs=40,
        batch_size=64,
        learning_rate=0.0005
    )
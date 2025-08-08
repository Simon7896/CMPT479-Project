import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import json
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from .gcn import GCNModel, GCNConfig
from .dataset import VulnerabilityDataset, create_data_splits


class GCNTrainer:
    """Trainer class for GCN model."""
    
    def __init__(self, config: GCNConfig):
        self.config = config
        self.device = config.device
        
        # Create directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize model
        self.model = GCNModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout
        ).to(self.device)
        
        # Apply Kaiming He initialization
        self._init_weights()
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Enhanced learning rate scheduler - more sensitive to validation accuracy
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # track validation accuracy instead of loss
            factor=0.5,
            patience=5,  # slightly more patient
            min_lr=1e-6
        )
        print("Using ReduceLROnPlateau scheduler with max mode")
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        print(f"Model initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _init_weights(self):
        """Initialize model weights using Kaiming He initialization."""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # He initialization for weights in convolutional and linear layers
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)
        print("Applied Kaiming He initialization to model weights")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model for one epoch."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                total_correct += (pred == batch.y).sum().item()
                total_samples += batch.y.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Early stopping patience: {self.config.patience}")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1:3d}/{self.config.num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                pred = out.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"Test Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return metrics
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        save_path = os.path.join(self.config.model_save_path, filename)
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model checkpoint."""
        load_path = os.path.join(self.config.model_save_path, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
        
        print(f"Model loaded from {load_path}")
    
    def plot_training_history(self, save_path: str | None = None):
        """Plot training history."""
        if not self.train_losses:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()


def prepare_data_loaders(data_dir: str, config: GCNConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepare data loaders for training, validation, and testing."""
    
    # Create dataset
    dataset = VulnerabilityDataset(data_dir)
    
    if len(dataset) == 0:
        raise ValueError(f"No valid graphs found in {data_dir}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset, 
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        test_ratio=config.test_split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    print(f"Data splits - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch


class GCNModel(nn.Module):
    """Graph Convolutional Network for vulnerability detection."""
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 384,  # back to larger size
                 num_layers: int = 5,    # increase layers slightly
                 num_classes: int = 2,
                 dropout: float = 0.15): # lower dropout - between previous values
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout

        # More effective architecture - alternating GCN and GAT
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer: GCN to process input features
                self.convs.append(GCNConv(input_dim, hidden_dim))
            elif i % 2 == 0:
                # Even layers: GCN
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                # Odd layers: GAT with 2 heads for better attention
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout))

        # Use batch normalization only
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Keep simple residual connections
        self.use_residual = True

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Apply graph layers with batch norm, dropout, and residuals
        for i, conv in enumerate(self.convs):
            # Store for residual connection
            prev_x = x
            # Apply convolution
            x = conv(x, edge_index)
            # Apply batch normalization
            x = self.batch_norms[i](x)
            # Apply activation
            x = F.relu(x)
            
            # Apply dropout except at the last layer
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection when shapes match
            if self.use_residual and i > 0 and x.shape == prev_x.shape:
                x = x + prev_x
            
        # Global pooling with both mean and max pooling (proven effective combo)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Concatenate the pooling results
        x_cat = torch.cat([x_mean, x_max], dim=1)
        # Classification
        out = self.classifier(x_cat)
        return out


class GCNConfig:
    """Configuration class for GCN model."""
    
    def __init__(self):
        # Model parameters 
        self.input_dim = 128
        self.hidden_dim = 384
        self.num_layers = 5
        self.num_classes = 2
        self.dropout = 0.15

        # Training parameters
        self.learning_rate = 0.0003
        self.weight_decay = 3e-4
        self.batch_size = 32
        self.num_epochs = 100
        self.patience = 15

        # Data parameters
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths
        self.model_save_path = './model/checkpoints/'
        self.log_dir = './model/logs/'

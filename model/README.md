# GCN Vulnerability Detection Model

This directory contains a Graph Convolutional Network (GCN) implementation for vulnerability detection in code graphs.

## Overview

The model uses PyTorch Geometric to implement a GCN that classifies graphs as vulnerable or non-vulnerable. It includes:

- **GCN Model**: Multi-layer graph convolutional network with batch normalization and dropout
- **Dataset Loader**: Loads and processes JSON graph files from the outputs directory
- **Training Pipeline**: Complete training loop with validation and early stopping
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## File Structure

```
model/
├── __init__.py          # Package initialization
├── gcn.py              # GCN model and configuration
├── dataset.py          # Dataset loading and preprocessing
└── trainer.py          # Training and evaluation logic

train_gcn.py            # Main training script
example_gcn.py          # Example usage
requirements.txt        # Dependencies
```

## Usage

### Quick Start

Run the example script to see how the model works:

```bash
python example_gcn.py
```

### Training the Model

Train the GCN model on your graph data:

```bash
python train_gcn.py --data_dir ./outputs --epochs 100 --batch_size 32
```

### Command Line Arguments

- `--data_dir`: Directory containing JSON graph files (default: ./outputs)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--num_layers`: Number of GCN layers (default: 3)
- `--dropout`: Dropout rate (default: 0.2)
- `--patience`: Early stopping patience (default: 10)
- `--save_plots`: Save training plots

### Example with Custom Parameters

```bash
python train_gcn.py \
  --data_dir ./outputs \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --hidden_dim 128 \
  --num_layers 4 \
  --dropout 0.3 \
  --save_plots
```

## Model Architecture

The GCN model consists of:

1. **Input Layer**: Takes node features (default 128 dimensions)
2. **GCN Layers**: Multiple graph convolutional layers with batch normalization and ReLU activation
3. **Global Pooling**: Aggregates node features to graph-level representation
4. **Classification Head**: Two-layer MLP for binary classification

## Data Format

The model expects JSON files in the outputs directory with the following structure:

```json
{
  "nodes": [
    {
      "id": 0,
      "type": "function_call",
      "value": "malloc",
      ...
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "type": "data_flow"
    }
  ]
}
```

## Configuration

Modify the `GCNConfig` class in `model/gcn.py` to adjust default parameters:

```python
class GCNConfig:
    def __init__(self):
        # Model parameters
        self.input_dim = 128
        self.hidden_dim = 256
        self.num_layers = 3
        self.num_classes = 2
        self.dropout = 0.2
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.patience = 10
```

## Output

The training script saves:

- **Model checkpoints**: Best and final model weights
- **Training logs**: Loss and accuracy history
- **Evaluation metrics**: Test set performance
- **Training plots**: Loss and accuracy curves (if --save_plots is used)

## Customization

### Adding New Features

To add new node features, modify the `_extract_node_features` method in `dataset.py`:

```python
def _extract_node_features(self, node):
    features = [0.0] * 128
    
    # Add your feature extraction logic here
    if 'new_feature' in node:
        features[10] = node['new_feature']
    
    return features
```
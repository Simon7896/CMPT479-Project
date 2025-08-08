#!/usr/bin/env python3
"""
Example script showing how to use the GCN model for vulnerability detection.
"""

import torch
from model import GCNConfig, GCNTrainer, VulnerabilityDataset, prepare_data_loaders


def example_training():
    """Example of how to train the model."""
    
    # Create configuration
    config = GCNConfig()
    config.num_epochs = 5  # Small number for quick demo
    config.batch_size = 16
    config.patience = 3
    
    print("Example GCN Training")
    print("=" * 50)
    
    try:
        # Check if data directory exists
        data_dir = './outputs'
        print(f"Looking for data in: {data_dir}")
        
        # Prepare data loaders
        train_loader, val_loader, test_loader = prepare_data_loaders(data_dir, config)
        
        # Initialize trainer
        trainer = GCNTrainer(config)
        
        # Train model
        print("\nStarting training...")
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = trainer.evaluate(test_loader)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nThis might be because:")
        print("1. No JSON files found in ./outputs directory")
        print("2. JSON files don't have the expected structure")
        print("3. Missing dependencies (run: pip install -r requirements.txt)")


def example_dataset_inspection():
    """Example of how to inspect the dataset."""
    
    print("Dataset Inspection Example")
    print("=" * 50)
    
    try:
        # Create dataset
        dataset = VulnerabilityDataset('./outputs')
        
        print(f"Total graphs in dataset: {len(dataset)}")
        
        if len(dataset) > 0:
            # Get first graph
            sample_graph = dataset[0]
            print(f"\nSample graph:")
            print(f"  Number of nodes: {sample_graph.x.shape[0]}")
            print(f"  Number of edges: {sample_graph.edge_index.shape[1]}")
            print(f"  Node feature dim: {sample_graph.x.shape[1]}")
            print(f"  Label: {sample_graph.y.item()}")
        else:
            print("No graphs found in dataset")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    print("GCN Vulnerability Detection Model")
    print("=" * 50)
    
    # Check if PyTorch is available
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Run examples
    example_dataset_inspection()
    print()
    example_training()

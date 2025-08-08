#!/usr/bin/env python3
"""
Training script for GCN vulnerability detection model.
"""

import argparse
import os
import json
import torch
from model import GCNConfig, GCNTrainer, prepare_data_loaders


def main():
    parser = argparse.ArgumentParser(description='Train GCN model for vulnerability detection')
    parser.add_argument('--data_dir', type=str, default='./outputs', 
                       help='Directory containing JSON graph files')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0003, 
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, 
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, 
                       help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.2, 
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, 
                       help='Early stopping patience')
    parser.add_argument('--save_plots', action='store_true', 
                       help='Save training plots')
    
    args = parser.parse_args()
    
    # Create configuration
    config = GCNConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.num_layers
    config.dropout = args.dropout
    config.patience = args.patience
    
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Prepare data loaders
        print("Preparing data loaders...")
        train_loader, val_loader, test_loader = prepare_data_loaders(args.data_dir, config)
        
        # Initialize trainer
        trainer = GCNTrainer(config)
        
        # Train model
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        
        # Save results
        def make_serializable(obj):
            """Convert non-serializable objects to serializable format."""
            if hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, torch.dtype):
                return str(obj)
            elif isinstance(obj, type):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        results = {
            'config': make_serializable(config.__dict__),
            'training_history': history,
            'test_metrics': test_metrics
        }
        
        results_path = os.path.join(config.log_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        # Save plots if requested
        if args.save_plots:
            plot_path = os.path.join(config.log_dir, 'training_history.png')
            trainer.plot_training_history(plot_path)
        
        # Save final model
        trainer.save_model('final_model.pth')
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

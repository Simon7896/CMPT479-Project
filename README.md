# Graph Neural Networks for Vulnerability Detection

This project implements a Graph Neural Network (GNN) approach for vulnerability prediction in C/C++ source code. The system extracts Abstract Syntax Trees (ASTs) and Control Flow Graphs (CFGs) from source functions using Clang LibTooling, then trains a hybrid GCN-GAT model to classify functions as vulnerable or non-vulnerable.

## Features

- **Graph Extraction**: AST and CFG extraction from C/C++ functions using Clang LibTooling
- **Hybrid GNN Model**: Combined GCN-GAT architecture with attention mechanisms
- **Complete Pipeline**: Data processing, training, evaluation, and visualization
- **Docker Support**: GPU-enabled containerized environment for reproducible experiments
- **Evaluation Tools**: Comprehensive metrics, confusion matrices, and ROC curves

## Requirements

- **Hardware**: NVIDIA GPU (recommended for training)
- **Software**: Docker with NVIDIA Container Toolkit, or native Ubuntu 22.04 setup
- **Dataset**: NIST Juliet Test Suite for C/C++ (included in data processing)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Simon7896/CMPT479-Project.git
cd CMPT479-Project
```

### 2. Docker Setup (Recommended)

Build the Docker image with CUDA support:
```bash
docker-compose build
```

Start the container:
```bash
docker-compose run --rm gcn-training bash
```

Or use the convenience script:
```bash
./run_docker_gpu.sh
```

### 3. Data Processing (Inside Container)

Extract and process the Juliet dataset:
```bash
cd data
# Extract the existing dataset
unzip -q 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip

# Build the parsing tool
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Process all C/C++ functions (this takes several minutes)
cd ..
./process_all_parallel_v2.sh
```

### 4. Training

Train the GNN model:
```bash
# Quick test run (5 epochs)
./run_training.sh --quick-test

# Full training run
./run_training.sh --epochs 100 --save-plots

# Custom configuration
./run_training.sh --epochs 50 --batch-size 16 --learning-rate 0.001
```

### 5. Generate Visualizations

Create plots from training results:
```bash
# Generate training curves and confusion matrix
python3 generate_plots.py

# Include ROC curve (requires evaluation)
python3 generate_plots.py --eval-roc --data_dir data/outputs --ckpt best
```

## Usage

### Training Options

The `run_training.sh` script supports various options:

```bash
./run_training.sh [OPTIONS]

Options:
  -d, --data-dir DIR        Directory containing JSON graph files (default: ./data/outputs)
  -e, --epochs NUM          Number of training epochs (default: 100)
  -b, --batch-size NUM      Batch size for training (default: 32)
  -l, --learning-rate RATE  Learning rate (default: 0.0003)
  -h, --hidden-dim NUM      Hidden dimension size (default: 384)
  -n, --num-layers NUM      Number of GCN layers (default: 5)
  -p, --dropout RATE        Dropout rate (default: 0.15)
  -t, --patience NUM        Early stopping patience (default: 15)
  -s, --save-plots          Enable plot generation guidance
  -q, --quick-test          Quick test with 5 epochs
```

### Direct Python Training

```bash
python3 train_gcn.py --data_dir data/outputs --epochs 100 --batch_size 32
```

### Visualization and Analysis

```bash
# Generate all plots from saved training logs
python3 generate_plots.py

# Generate plots with ROC curve evaluation
python3 generate_plots.py --eval-roc --data_dir data/outputs --ckpt best

# Use custom training results
python3 generate_plots.py --log-json custom_results.json
```

## Native Installation (Alternative)

If not using Docker:

### Dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake ninja-build python3 python3-pip \
    llvm clang clang-tools libclang-dev llvm-dev \
    libedit-dev libncurses5-dev zlib1g-dev libxml2-dev
```

### Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## File Structure

```
CMPT479-Project/
├── data/                          # Dataset and processing
│   ├── 2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
│   ├── main.cpp                   # Clang LibTooling parser
│   ├── process_all_parallel_v2.sh # Data processing script
│   ├── build/                     # Build directory for parser
│   └── outputs/                   # Processed JSON graphs
├── model/                         # ML model and training
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Geometric dataset
│   ├── gcn.py                     # GNN model architecture
│   ├── trainer.py                 # Training and evaluation
│   ├── checkpoints/               # Saved model weights
│   └── logs/                      # Training logs and plots
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile                     # Docker image definition
├── run_docker_gpu.sh              # GPU Docker convenience script
├── run_training.sh                # Training script with options
├── train_gcn.py                   # Main training script
├── generate_plots.py              # Visualization generation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Output Files

After training and visualization:

- `model/checkpoints/best_model.pth` - Best performing model weights
- `model/checkpoints/final_model.pth` - Final epoch model weights  
- `model/logs/training_results.json` - Complete training metrics and history
- `model/logs/training_loss.png` - Training/validation loss curves
- `model/logs/training_accuracy.png` - Training/validation accuracy curves
- `model/logs/confusion_matrix.png` - Test set confusion matrix
- `model/logs/roc_curve.png` - ROC curve with AUC score

## Troubleshooting

### GPU Support
Ensure NVIDIA Container Toolkit is installed:
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Memory Issues
For large datasets, reduce batch size:
```bash
./run_training.sh --batch-size 16 --epochs 50
```

### Data Processing
If data processing fails, ensure sufficient disk space and try processing smaller batches in `process_all_parallel_v2.sh`.
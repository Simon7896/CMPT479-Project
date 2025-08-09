#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_DIR="./outputs"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.0003
HIDDEN_DIM=384
NUM_LAYERS=5
DROPOUT=0.15
PATIENCE=15
SAVE_PLOTS=false
QUICK_TEST=false

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Train GCN model for vulnerability detection"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir DIR        Directory containing JSON graph files (default: ./outputs)"
    echo "  -e, --epochs NUM          Number of training epochs (default: 100)"
    echo "  -b, --batch-size NUM      Batch size for training (default: 32)"
    echo "  -l, --learning-rate RATE  Learning rate (default: 0.001)"
    echo "  -h, --hidden-dim NUM      Hidden dimension size (default: 256)"
    echo "  -n, --num-layers NUM      Number of GCN layers (default: 3)"
    echo "  -p, --dropout RATE        Dropout rate (default: 0.2)"
    echo "  -t, --patience NUM        Early stopping patience (default: 10)"
    echo "  -s, --save-plots          Save training plots"
    echo "  -q, --quick-test          Quick test with minimal epochs (5 epochs, small batch)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Train with default settings"
    echo "  $0 --quick-test                 # Quick test run"
    echo "  $0 -e 50 -b 16 --save-plots     # 50 epochs, batch size 16, save plots"
    echo "  $0 -d ./my_data -e 200          # Custom data directory, 200 epochs"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -h|--hidden-dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        -n|--num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        -p|--dropout)
            DROPOUT="$2"
            shift 2
            ;;
        -t|--patience)
            PATIENCE="$2"
            shift 2
            ;;
        -s|--save-plots)
            SAVE_PLOTS=true
            shift
            ;;
        -q|--quick-test)
            QUICK_TEST=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Quick test settings
if [[ "$QUICK_TEST" == "true" ]]; then
    EPOCHS=5
    BATCH_SIZE=16
    PATIENCE=3
    SAVE_PLOTS=true
    echo -e "${YELLOW}Quick test mode enabled:${NC}"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Patience: $PATIENCE"
    echo ""
fi

# Print configuration
echo -e "${BLUE}GCN Training Configuration:${NC}"
echo "  Data directory: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Hidden dimension: $HIDDEN_DIM"
echo "  Number of layers: $NUM_LAYERS"
echo "  Dropout: $DROPOUT"
echo "  Patience: $PATIENCE"
echo "  Save plots: $SAVE_PLOTS"
echo ""

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    echo -e "${RED}Error: Data directory '$DATA_DIR' does not exist${NC}"
    exit 1
fi

# Check if there are JSON files in the data directory
json_count=$(find "$DATA_DIR" -name "*.json" | wc -l)
if [[ $json_count -eq 0 ]]; then
    echo -e "${YELLOW}Warning: No JSON files found in '$DATA_DIR'${NC}"
    echo "Make sure to run the data processing pipeline first."
    echo ""
fi

# Check if Python dependencies are installed
echo -e "${BLUE}Checking dependencies...${NC}"
python3 -c "
import torch
import torch_geometric
print(f'PyTorch: {torch.__version__}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
" 2>/dev/null || {
    echo -e "${RED}Error: Missing dependencies. Please install them:${NC}"
    echo "pip install -r requirements.txt"
    exit 1
}

echo ""

# Create directories if they don't exist
mkdir -p ./model/checkpoints
mkdir -p ./model/logs

# Build command
cmd="python3 train_gcn.py"
cmd+=" --data_dir $DATA_DIR"
cmd+=" --epochs $EPOCHS"
cmd+=" --batch_size $BATCH_SIZE"
cmd+=" --learning_rate $LEARNING_RATE"
cmd+=" --hidden_dim $HIDDEN_DIM"
cmd+=" --num_layers $NUM_LAYERS"
cmd+=" --dropout $DROPOUT"
cmd+=" --patience $PATIENCE"

if [[ "$SAVE_PLOTS" == "true" ]]; then
    cmd+=" --save_plots"
fi

# Start training
echo -e "${GREEN}Starting GCN training...${NC}"
echo "Command: $cmd"
echo ""

# Run the training
eval $cmd

# Check if training was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo ""
    echo "Generated files:"
    echo "  Model checkpoints: ./model/checkpoints/"
    echo "  Training logs: ./model/logs/"
    if [[ "$SAVE_PLOTS" == "true" ]]; then
        echo "  To generate plots from logs, run:"
        echo "    python3 generate_plots.py --eval-roc --data_dir $DATA_DIR --ckpt best"
        echo "  Plots will be saved to:"
        echo "    ./model/logs/training_loss.png"
        echo "    ./model/logs/training_accuracy.png"
        echo "    ./model/logs/confusion_matrix.png"
        echo "    ./model/logs/roc_curve.png (if available)"
    fi
else
    echo ""
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

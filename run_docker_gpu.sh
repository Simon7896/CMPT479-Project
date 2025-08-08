#!/bin/bash

# Script to run Docker container with GPU support

echo "Starting Docker container with NVIDIA GPU support..."

docker run -it --rm \
    --gpus all \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v $(pwd):/app \
    -w /app \
    gcn-training \
    bash

echo "Container started with GPU support"

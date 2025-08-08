# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    python3-setuptools \
    llvm \
    clang \
    clang-tools \
    libclang-dev \
    llvm-dev \
    libedit-dev \
    libncurses5-dev \
    zlib1g-dev \
    libxml2-dev \
    libssl-dev \
    pkg-config \
    lsb-release \
    ca-certificates \
    openjdk-19-jdk \
    software-properties-common \
    sudo \
    parallel && \
    rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy source files into the container
# Assuming main.cpp and CMakeLists.txt are in the same directory
COPY . .

# Upgrade pip and install all dependencies
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt --resume-retries 3

# Download dataset
RUN cd data && \
wget https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip

# Build the tool
RUN mkdir -p build && cd build && \
    cmake ../data && \
    make -j$(nproc)

# Default command
CMD ["/bin/bash"]


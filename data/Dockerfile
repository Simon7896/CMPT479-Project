FROM ubuntu:22.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

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
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Optional: upgrade pip and install PyTorch Geometric dependencies
RUN pip3 install --upgrade pip && \
    pip3 install torch torch_geometric --resume-retries 3

# Set working directory
WORKDIR /app

# Copy your source files into the container
# Assuming you have main.cpp and CMakeLists.txt in the same directory
COPY . .

# Build the tool
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Default command (can be overridden)
CMD ["/bin/bash"]


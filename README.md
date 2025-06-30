# CMPT479-Project

## Description

## Table of Contents

1. [Description](#description)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)

## Requirements

- Ubuntu 22.04
- LLVM/Clang 14
- CMake, Ninja
- Python 3
- Pytorch, Pytorch Geometric

# Installation

## 1. Clone the Repo

```bash
git clone https://github.com/Simon7896/CMPT479-Project.git
cd CMPT479-Project
```

## 2a. Docker Setup
### 3a. Build Docker Image (If using docker)
```bash
docker build -t cmpmt479-project .
```

### 4a. Run interactive container
```bash
docker run -it cmpt479-project@latest --name cmpt479-project-container
```

## 2b. Install dependencies (If not using docker)

```bash
apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    python3-setuptools \
    unzip \
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
    software-properties-common
```

### 3b. Download and unzip Juliet dataset
```bash
cd data &&
wget https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip
```

### 4b. Build the AST+CFG parsing tool
```bash
mkdir -p build && cd build && \
    cmake ../data && \
    make -j$(nproc)
```

### 5b. (Optional) create and activate python venv
```
python3 -m venv .venv &&
source .venv/bin/activate
```

### 6b. Install python dependencies
```bash
pip install --upgrade pip &&
pip install torch torch_geometric
```




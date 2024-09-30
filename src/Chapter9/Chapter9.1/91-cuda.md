# CUDA

```bash
#!/bin/bash

# 步骤 1: 更新系统并安装必要的依赖
echo "更新系统并安装必要依赖..."
sudo apt-get update
sudo apt-get install -y build-essential dkms

# 步骤 2: 添加 NVIDIA CUDA 工具包存储库
echo "添加 NVIDIA CUDA 存储库..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# 步骤 3: 安装 CUDA 工具包
echo "安装 CUDA 工具包..."
sudo apt-get update
sudo apt-get -y install cuda

# 步骤 4: 设置环境变量
echo "配置 CUDA 环境变量..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 步骤 5: 验证 CUDA 安装
echo "验证 CUDA 安装..."
nvcc --version

echo "CUDA 安装完成。请重新启动系统以确保更改生效。"
```

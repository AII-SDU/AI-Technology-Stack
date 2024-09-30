# Triton (NVIDIA)

```bash
#!/bin/bash

# 步骤 1: 更新系统
echo "更新系统..."
sudo apt-get update && sudo apt-get upgrade -y

# 步骤 2: 检查 CUDA 是否安装
echo "检查 CUDA 安装情况..."
if ! command -v nvcc &> /dev/null
then
    echo "CUDA 未安装，请先安装 CUDA。"
    exit 1
else
    nvcc --version
fi

# 步骤 3: 检查 NVIDIA 驱动是否安装
echo "检查 NVIDIA 驱动安装情况..."
if ! command -v nvidia-smi &> /dev/null
then
    echo "NVIDIA 驱动未安装，请先安装驱动。"
    exit 1
else
    nvidia-smi
fi

# 步骤 4: 安装 Python 3.9 和 pip
echo "安装 Python 3.9 和 pip..."
sudo apt-get install -y python3 python3-pip

# 步骤 5: 安装 PyTorch (CUDA 11.8 版本)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 步骤 6: 安装 Triton
echo "安装 Triton..."
pip install triton

# 验证 Triton 安装
echo "验证 Triton 安装..."
python3 -c "import triton; print(triton.__version__)"

echo "Triton 安装完成。"
```

### 脚本说明：
1. **系统更新**：首先更新系统软件包并升级现有软件。
2. **检查CUDA和驱动**：检查系统是否已安装CUDA和NVIDIA驱动，如果未安装则终止脚本执行。
3. **安装Python 3.9**：通过APT安装Python 3.9及其对应的pip工具。
4. **安装PyTorch**：通过`pip`安装支持CUDA 11.8的PyTorch及相关库。
5. **安装Triton**：通过`pip`安装Triton库。
6. **验证Triton安装**：通过Python命令检查Triton版本以确保安装成功。

此脚本将自动执行所有步骤，确保环境配置正确并安装Triton。
# Apache TVM (AMD)

```bash
#!/bin/bash

# 步骤 1: 安装基本依赖
echo "安装基本依赖..."
sudo apt update
sudo apt install -y git cmake build-essential libtinfo-dev zlib1g-dev \
python3-dev python3-setuptools python3-pip

# 步骤 2: 安装 LLVM
echo "安装 LLVM..."
sudo apt install -y llvm clang

# 步骤 3: 克隆 Apache TVM 源码
echo "克隆 Apache TVM 源码..."
git clone --recursive https://github.com/apache/tvm tvm
cd tvm

# 步骤 4: 安装 ROCm（适用于 AMD 显卡）
echo "添加 ROCm 仓库并安装 ROCm..."
sudo apt update
sudo apt install -y wget gnupg2
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.5 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 安装 ROCm
sudo apt update
sudo apt install -y rocm-dkms

# 步骤 5: 编译 Apache TVM
echo "开始编译 Apache TVM..."
cp cmake/config.cmake .

# 配置使用 LLVM 和 ROCm
sed -i '/set(USE_LLVM OFF)/c\set(USE_LLVM llvm-config)' cmake/config.cmake
sed -i '/set(USE_ROCM OFF)/c\set(USE_ROCM ON)' cmake/config.cmake

# 创建构建目录并编译
mkdir build
cd build
cmake ..
make -j$(nproc)

# 步骤 6: 设置 Python 环境
echo "设置 Python 环境..."
cd ../python
sudo python3 setup.py install

# 步骤 7: 验证安装
echo "验证 TVM 安装..."
python3 -c "import tvm; print('TVM 安装成功')"

echo "Apache TVM 安装完成。"
```

### 脚本说明：
1. **安装基本依赖**：首先安装构建TVM所需的基本工具，如Git、CMake和Python开发库。
2. **安装LLVM**：TVM依赖LLVM进行编译和优化。
3. **克隆TVM源码**：克隆官方Apache TVM的源码。
4. **安装ROCm**：适用于AMD显卡的支持，添加ROCm仓库并安装驱动。
5. **编译TVM**：将`USE_LLVM`设置为`llvm-config`，启用`USE_ROCM`支持，并编译TVM。
6. **设置Python环境**：安装Python相关的TVM库，确保TVM可以通过Python调用。
7. **验证安装**：通过导入`tvm`模块验证TVM是否正确安装。

运行该脚本时，请根据系统兼容性确保步骤配置正确，特别是ROCm部分。
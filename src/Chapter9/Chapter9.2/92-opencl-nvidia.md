# OpenCL (NVIDIA)
```bash
#!/bin/bash

# OpenCL ICD安装步骤
echo "开始安装OpenCL ICD..."

# 检查NVIDIA驱动是否已安装
echo "检查NVIDIA驱动安装情况..."
nvidia-smi

# 更新软件包列表
echo "更新软件包列表..."
sudo apt update

# 安装OpenCL ICD和NVIDIA OpenCL开发库
echo "安装OpenCL ICD和NVIDIA OpenCL开发库..."
sudo apt install -y ocl-icd-libopencl1 nvidia-opencl-dev

# 安装clinfo并检查OpenCL安装情况
echo "安装clinfo工具并检查OpenCL平台和设备信息..."
sudo apt-get install -y clinfo
clinfo

# 检查OpenCL头文件是否安装成功
echo "检查OpenCL头文件..."
ls /usr/include/CL

echo "OpenCL ICD安装完成。"

# OpenCL Runtime安装步骤
echo "开始安装OpenCL Runtime..."

# 检查NVIDIA驱动是否已安装
echo "再次检查NVIDIA驱动安装情况..."
nvidia-smi

# 更新软件包列表
echo "更新软件包列表..."
sudo apt update

# 安装CUDA工具包（需要根据操作系统版本替换<distro>和<version>）
echo "安装CUDA工具包..."
CUDA_REPO_PKG="cuda-repo-<distro>_<version>_amd64.deb"
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda

# 配置环境变量
echo "配置CUDA环境变量..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 安装clinfo并再次检查OpenCL安装情况
echo "再次检查OpenCL平台和设备信息..."
sudo apt-get install -y clinfo
clinfo

echo "OpenCL Runtime安装完成。"

# OpenCL C/C++开发环境安装步骤
echo "开始安装OpenCL C/C++开发环境..."

# 检查NVIDIA驱动是否已安装
echo "检查NVIDIA驱动安装情况..."
nvidia-smi

# 更新软件包列表
echo "更新软件包列表..."
sudo apt update

# 安装CUDA工具包
echo "安装CUDA工具包..."
sudo dpkg -i ${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install -y cuda

# 配置环境变量
echo "配置CUDA环境变量..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 安装OpenCL开发库
echo "安装OpenCL开发库..."
sudo apt-get install -y ocl-icd-opencl-dev

# 测试OpenCL C/C++开发环境（可选：运行C++示例代码）
echo "OpenCL C/C++开发环境安装完成。"
```

在运行脚本之前，请确保将 `<distro>` 和 `<version>` 替换为实际的操作系统版本和CUDA版本。
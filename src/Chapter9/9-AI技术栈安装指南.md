# 附录——AI技术栈安装指南 
### OpenCL ICD安装
1. 检验驱动是否安装 （需要先安装驱动）
nvidia-smi
2. 更新软件包列表
sudo apt update
3. 安装OpenCL ICD和Nvidia OpenCL库
sudo apt install ocl-icd-libopencl1 nvidia-opencl-dev；
注：ocl-icd-libopencl1是OpenCL的安装时可发现（ICD）管理器，它允许系统发现并使用多个OpenCL实现；nvidia-opencl-dev包含了Nvidia的OpenCL开发文件；
4. 检验是否安装成功
sudo apt-get install clinfo
clinfo
注意：OpenCL Runtime安装成功，clinfo 将列出所有可用的OpenCL平台和设备信息，包括平台名称、设备类型（如GPU）、支持的OpenCL版本等
或者
4. 检验OpenCL是否安装成功，如果列出了OpenCL的头文件（如cl.h、cl_ext.h等），那么说明头文件已经安装
ls /usr/include/CL



### OpenCL Runtime 安装
对于NVIDIA GPU，CUDA Toolkit中包含了OpenCL的支持。因此，安装CUDA Toolkit通常会自动安装OpenCL Runtime
1. 检验驱动是否安装 （需要先安装驱动）
nvidia-smi；
2. 更新软件包列表
sudo apt update
3. 安装CUDA工具包
sudo dpkg -i cuda-repo-<distro>_<version>_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
4. 配置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
5. 检验是否安装成功
sudo apt-get install clinfo
clinfo
注意：OpenCL Runtime安装成功，clinfo 将列出所有可用的OpenCL平台和设备信息，包括平台名称、设备类型（如GPU）、支持的OpenCL版本等



### OpenCL C/C++ 安装
1. 检验驱动是否安装 （需要先安装驱动）
nvidia-smi；
2. 更新软件包列表
sudo apt update
3. 安装CUDA工具包
sudo dpkg -i cuda-repo-<distro>_<version>_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/<distro>/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
4. 配置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
5. 安装OpenCL开发库
sudo apt-get install ocl-icd-opencl-dev
6. 测试OpenCL C/C++环境 
需要示例代码（C++）


### Triton （paddle_benchmark环境）
Triton version: 3.0.0
1. 更新系统
sudo apt-get update
sudo apt-get upgrade
2. 检验CUDA是否安装
nvcc --version
3. 检验驱动是否安装
nvidia-smi
4. 安装Python建议3.9版本
sudo apt-get install python3 python3-pip
5. 安装pythorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
6. 安装 Triton
pip install triton
检验是否安装成功python3 -c "import triton; print(triton.__version__)"


Ubuntu系统中 NVIDIA 4080s显卡 安装Apache_TVM
1.安装依赖项
sudo apt-get update
sudo apt-get install -y git cmake build-essential libtinfo-dev zlib1g-dev \
                        libcurl4-openssl-dev libopenblas-dev python3-dev \
                        python3-pip python3-setuptools python3-venv
2.克隆 Apache TVM 的 GitHub 仓库
git clone --recursive https://github.com/apache/tvm.git
cd tvm
3.安装 LLVM
sudo apt-get install -y llvm llvm-dev clang
4.配置 TVM
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_LLVM=ON -DCMAKE_BUILD_TYPE=Release
5.构建 TVM
make -j$(nproc)
6.设置环境变量，在 ~/.bashrc 文件中添加环境变量
echo 'export TVM_HOME=~/tvm' >> ~/.bashrc
echo 'export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}' >> ~/.bashrc
echo 'export PATH=$TVM_HOME/build:${PATH}' >> ~/.bashrc
source ~/.bashrc
7.安装 Python 依赖项
pip install numpy
pip install -e ${TVM_HOME}/python
## 如果缺少.so可以运行如下代码进行解决
cd ${TVM_HOME}
mkdir -p build
cp cmake/config.cmake build
cd build
cmake ..
make -j$(nproc)
8.验证安装
import tvm
print("TVM version:", tvm.__version__)
## 如果出现发现缺少所需的 libstdc++.so.6 的 GLIBCXX_3.4.30 版本的情况可尝试运行如下代码
ldd /home/aii-works/anaconda3/envs/paddle_benchmark/bin/python | grep libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6


## openaCC的安装
1.下载 解压 安装NVIDIA HPC SDK

网址：https://developer.nvidia.com/
wget https://developer.download.nvidia.com/hpc-sdk/24.7/nvhpc_2024_247_Linux_x86_64_cuda_multi.tar.gztar xpzf nvhpc_2024_247_Linux_x86_64_cuda_multi.tar.gznvhpc_2024_247_Linux_x86_64_cuda_multi/install
## 默认路径会安装到 /opt/nvidia/hpc_sdk
2.添加环境变量
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/bin:$PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/<version>/compilers/lib:$LD_LIBRARY_PATH
## 注意将<version>修改为安装的HPC SDK版本，本次安装版本为24.7
3.测试程序

```C++
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#define N 10000

// 矩阵加法函数：CPU 版本
void matrix_add_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

// 矩阵加法函数：GPU 版本
void matrix_add_gpu(float *A, float *B, float *C, int n) {
    #pragma acc parallel loop collapse(2) copyin(A[0:n*n], B[0:n*n]) copyout(C[0:n*n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

int main() {
    float *A, *B, *C;
    A = (float*) malloc(N * N * sizeof(float));
    B = (float*) malloc(N * N * sizeof(float));
    C = (float*) malloc(N * N * sizeof(float));

    // 初始化矩阵
    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // 记录 CPU 时间
    clock_t start_cpu = clock();
    matrix_add_cpu(A, B, C, N);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\n", cpu_time);

    // 记录 GPU 时间
    clock_t start_gpu = clock();
    matrix_add_gpu(A, B, C, N);
    clock_t end_gpu = clock();
    double gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    printf("GPU Time: %f seconds\n", gpu_time);

    // 清理
    free(A);
    free(B);
    free(C);

    return 0;
}

```
4． 编译和运行代码
pgcc -acc -Minfo=accel -o matrix_add matrix_add.c
./matrix_add

## OpenCL安装
1.添加 ROCm (Radeon Open Compute) Repository，ROCm 包含了 AMD GPU 驱动程序以及 OpenCL 支持；
更新系统软件包：
sudo apt update
sudo apt upgrade

添加 ROCm 的官方软件源：
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

更新软件包索引：	
sudo apt update

2.安装 ROCm 和 OpenCL Runtime
sudo apt install rocm-dkms
sudo apt install rocm-opencl rocm-opencl-dev

3.设置环境变量
nano ~/.bashrc
在文件末尾添加以下内容：
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
保存文件并使其生效：
source ~/.bashrc

4.验证 OpenCL 安装
clinfo

5.安装必要的依赖库（可选）
sudo apt install ocl-icd-libopencl1 opencl-headers clinfo

## 安装Triton（本机安装）
在 Ubuntu 系统上为 AMD 显卡安装 Triton，通常涉及以下几个步骤。Triton 是一个高度优化的编译器，用于深度学习的矩阵乘法内核，主要针对 NVIDIA GPU，但你可以尝试使用它的部分功能或在特定情况下将其与 AMD GPU 配合使用。以下步骤将帮助你安装 Triton 和所需的依赖项。
1.安装必要依赖
sudo apt update
sudo apt install python3 python3-pip clang

2.安装 ROCm
添加 ROCm 软件源并安装 ROCm：
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms rocm-dev rocm-opencl

设置环境变量：
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

3.安装 Triton
pip install triton
## 需要提前安装pip install pybind11==2.13.1

## 安装Apache TVM
1.安装基本依赖
sudo apt update
sudo apt install -y git cmake build-essential libtinfo-dev zlib1g-dev \
python3-dev python3-setuptools python3-pip
2.安装LLVM
sudo apt install -y llvm clang
3.克隆Apache TVM源码
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
4.安装ROCm（适用于AMD显卡）
## 添加ROCm仓库
sudo apt update
sudo apt install -y wget gnupg2
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.5 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

## 安装ROCm
sudo apt update
sudo apt install -y rocm-dkms
5.编译Apache TVM
cp cmake/config.cmake .

set(USE_LLVM llvm-config)
set(USE_ROCM ON)

mkdir build
cd build
cmake ..
make -j$(nproc)
6.设置Python环境
cd ../python
sudo python3 setup.py install
7.验证安装
import tvm

## Intel 驱动安装
https://github.com/intel/intel-extension-for-tensorflow/issues/54

1.为 OpenGL 和 Vulkan 配置开源 Mesa 3d 图形库
sudo add-apt-repository ppa：oibaf/graphics-drivers
sudo apt update
sudo apt upgrade
dpkg -l |grep -i mesa
2.设置环境变量 PRIME，glmark2 基准测试中检查显卡的运行情况
export DRI_PRIME=1
glxinfo -B | grep -i device
glmark2
3.下载驱动的必须包
cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu-dbgsym_1.3.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd-dbgsym_23.30.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb

检查所有内容是否都已正确下载
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/ww30.sum
sha256sum -c ww30.sum

安装所有软件包
sudo dpkg -i *.deb
4.安装 oneAPI 库
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo “deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main” |sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update

sudo apt-get install intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh
sudo sh ./l_BaseKit_p_2023.2.0.49397_offline.sh
5.安装 TensorFlow 附加组件
sudo apt install pip
pip install --upgrade pip
pip install 'tensorflow==2.13.0'
installing the Intel add-on:
pip install --upgrade intel-extension-for-tensorflow[gpu]	
6.检查是否一切顺利
bash /path to site-packages/intel_extension_for_tensorflow/tools/env_check.sh
7.例子
启用oneAPI 运行环境
source /opt/intel/oneapi/setvars.sh

启用虚拟运行环境
source env_itex/bin/activate

运行如下例子
import numpy as np
import sys

import tensorflow as tf

## Conv + ReLU activation + Bias
N = 1
num_channel = 3
input_width, input_height = (5, 5)
filter_width, filter_height = (2, 2)

x = np.random.rand(N, input_width, input_height, num_channel).astype(np.float32)
weight = np.random.rand(filter_width, filter_height, num_channel, num_channel).astype(np.float32)
bias = np.random.rand(num_channel).astype(np.float32)

conv = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
activation = tf.nn.relu(conv)
result = tf.nn.bias_add(activation, bias)

print(result)
print('Finished')


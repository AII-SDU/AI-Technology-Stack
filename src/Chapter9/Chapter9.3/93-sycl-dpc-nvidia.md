# SYCL(NVIDIA)

```bash
#!/bin/bash

# 步骤 1: 安装必要依赖
echo "安装基本依赖..."
sudo apt update
sudo apt install -y cmake pkg-config build-essential

# 验证依赖是否安装成功
echo "验证依赖安装..."
which cmake pkg-config make gcc g++

# 步骤 2: 安装包含DPC++/C++编译器的英特尔oneAPI工具包版本2024.2.1
echo "下载并安装英特尔oneAPI工具包..."
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100_offline.sh
sudo sh ./l_BaseKit_p_2024.2.1.100_offline.sh -a --silent --cli --eula accept

# 步骤 3: 安装CUDA（假设已安装，可跳过该步骤）
echo "跳过CUDA安装步骤，假设已安装..."

# 步骤 4: 安装NVIDIA GPU插件
echo "下载并安装NVIDIA GPU插件..."
# 需要自行访问Codeplay下载页面下载插件，根据用户的DPC++/C++版本
wget https://developer.codeplay.com/products/oneapi/nvidia/download/oneapi-for-nvidia-gpus-2024.2.1-cuda-12.0-linux.sh
sudo sh oneapi-for-nvidia-gpus-2024.2.1-cuda-12.0-linux.sh

# 步骤 5: 设置环境变量
echo "设置环境变量..."

# 如果安装为系统范围的安装
echo "设置系统范围的环境变量..."
source /opt/intel/oneapi/setvars.sh --include-intel-llvm

# 如果安装为用户自定义位置，请替换路径为实际的安装路径
# echo "设置私人安装环境变量..."
# source ~/intel/oneapi/setvars.sh --include-intel-llvm

# 确保CUDA库和工具可用
echo "检查CUDA工具和库..."
nvidia-smi

# 如果nvidia-smi出现问题，手动设置CUDA路径
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 步骤 6: 验证安装
echo "验证SYCL安装..."
sycl-ls

# 验证结果
echo "如果输出显示类似以下内容：[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA A100-PCIE-40GB 8.0 [CUDA 12.5]，则安装成功。"
```

### 脚本说明：
1. **安装必要依赖**：更新系统并安装所需的构建工具，如CMake、pkg-config等。
2. **安装英特尔oneAPI工具包**：下载并以静默方式安装包含DPC++/C++编译器的英特尔oneAPI工具包。
3. **安装NVIDIA GPU插件**：下载并安装NVIDIA GPU的DPC++插件。
4. **设置环境变量**：配置英特尔工具链的环境变量，确保正确识别DPC++和CUDA路径。
5. **验证安装**：通过`sycl-ls`命令验证SYCL安装和NVIDIA GPU插件是否成功。

请确保替换脚本中的路径和版本信息以适应您具体的系统环境和安装需求。



## 计算库

以下是基于您提供的运行SYCL测试例子的自动化脚本。该脚本涵盖了DPC++安装、环境配置、项目克隆和编译的所有步骤。

```bash
#!/bin/bash

# 步骤 1: 安装 DPC++（Intel oneAPI）
echo "下载并安装 Intel oneAPI BaseKit..."
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100.sh
sudo sh ./l_BaseKit_p_2024.2.1.100.sh -a --silent --cli --eula accept

# 步骤 2: 设置环境变量
echo "设置 Intel oneAPI 环境变量..."
source /opt/intel/oneapi/setvars.sh

# 步骤 3: 配置 CMake 环境
echo "配置 CMake 环境..."
which icpx
export CXX=/opt/intel/oneapi/compiler/2024.2/bin/icpx

# 步骤 4: 克隆 portBLAS 仓库
echo "克隆 portBLAS 仓库..."
git clone https://github.com/codeplaysoftware/portBLAS.git
cd portBLAS

# 步骤 5: 设置临时环境变量
echo "设置临时环境变量..."
SYCL_LIB=$(find / -name libsycl.so.7 2>/dev/null)
export LD_LIBRARY_PATH=/opt/intel/oneapi/2024.2/lib:/opt/intel/oneapi/compiler/2024.2/lib:$LD_LIBRARY_PATH

# 步骤 6: 安装 GCC 11 版本
echo "安装 GCC 11 和相关依赖库..."
sudo apt install -y gcc-11 g++-11

# 步骤 7: 安装 OpenMP 和 OpenBLAS
echo "安装 OpenMP 和 OpenBLAS..."
sudo apt-get install -y libomp-dev libopenblas-dev

# 步骤 8: 编译 portBLAS
echo "开始编译 portBLAS..."
mkdir -p build && cd build
cmake -GNinja ../ -DSYCL_COMPILER=dpcpp -DDPCPP_SYCL_TARGET=nvptx64-nvidia-cuda -DDPCPP_SYCL_ARCH=sm_80 -DCMAKE_INSTALL_PREFIX=/usr/local/portBLAS

# 使用 Ninja 进行构建和安装
ninja
sudo ninja install

# 步骤 9: 运行示例程序
echo "运行 sample_gemm 示例程序..."
./samples/sample_gemm

# 如果遇到 CMake 配置错误，清理缓存后重新配置
if [ $? -ne 0 ]; then
  echo "CMake 配置失败，清理缓存并重新配置..."
  rm -rf CMakeCache.txt CMakeFiles
  cmake -GNinja ../ -DSYCL_COMPILER=dpcpp -DDPCPP_SYCL_TARGET=nvptx64-nvidia-cuda -DDPCPP_SYCL_ARCH=sm_80 -DCMAKE_INSTALL_PREFIX=/usr/local/portBLAS
  ninja
  sudo ninja install
fi

# 步骤 10: 编译自定义算子
echo "编译自定义 SYCL 代码..."
icpx -fsycl -o test_device ../samples/test.cpp
./test_device

echo "SYCL 测试完成。"
```

### 脚本说明：
1. **安装Intel oneAPI**：下载并以静默方式安装Intel oneAPI BaseKit。
2. **设置环境变量**：配置SYCL编译器的环境变量，确保DPC++正确使用。
3. **克隆portBLAS仓库**：从GitHub克隆portBLAS代码库。
4. **配置CMake和编译选项**：通过CMake进行配置，指定使用NVIDIA GPU并使用Ninja构建工具编译。
5. **运行示例**：编译并运行`sample_gemm`示例测试SYCL环境。
6. **编译自定义SYCL代码**：编译并运行自定义的SYCL程序`test_device`。


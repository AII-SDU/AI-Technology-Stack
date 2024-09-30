# oneAPI


```bash
#!/bin/bash

# 步骤 1: 为 OpenGL 和 Vulkan 配置开源 Mesa 3D 图形库
echo "配置 OpenGL 和 Vulkan 的 Mesa 3D 图形库..."
sudo add-apt-repository ppa:oibaf/graphics-drivers
sudo apt update
sudo apt upgrade -y

# 验证 Mesa 安装
dpkg -l | grep -i mesa

# 步骤 2: 设置环境变量以使用独立显卡
echo "设置环境变量..."
export DRI_PRIME=1
glxinfo -B | grep -i device
glmark2

# 步骤 3: 下载并安装 Intel GPU 必需的软件包
echo "下载并安装 Intel GPU 必需的软件包..."
mkdir -p neo && cd neo
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-core_1.0.14828.8_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.14828.8/intel-igc-opencl_1.0.14828.8_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu-dbgsym_1.3.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-level-zero-gpu_1.3.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd-dbgsym_23.30.26918.9_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/intel-opencl-icd_23.30.26918.9_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/libigdgmm12_22.3.0_amd64.deb

# 验证下载的包是否正确
wget https://github.com/intel/compute-runtime/releases/download/23.30.26918.9/ww30.sum
sha256sum -c ww30.sum

# 安装所有软件包
sudo dpkg -i *.deb
cd ..

# 步骤 4: 安装 oneAPI 库
echo "安装 oneAPI 库..."
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor --output /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-runtime-dpcpp-cpp intel-oneapi-runtime-mkl

# 安装 oneAPI BaseKit
echo "下载并安装 oneAPI BaseKit..."
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh
sudo sh ./l_BaseKit_p_2023.2.0.49397_offline.sh

# 步骤 5: 安装 TensorFlow 及其扩展
echo "安装 TensorFlow 及其扩展..."
sudo apt install -y python3-pip
pip install --upgrade pip
pip install 'tensorflow==2.13.0'
pip install --upgrade intel-extension-for-tensorflow[gpu]

# 步骤 6: 检查环境配置
echo "检查 oneAPI 和 TensorFlow 配置..."
bash $(python3 -c "import site; print(site.getsitepackages()[0])")/intel_extension_for_tensorflow/tools/env_check.sh

# 步骤 7: 运行示例代码
echo "运行 TensorFlow 示例代码..."
cat <<EOL > oneapi_tensorflow_example.py
import numpy as np
import tensorflow as tf

# Conv + ReLU activation + Bias
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
EOL

# 设置 oneAPI 环境并运行示例
source /opt/intel/oneapi/setvars.sh
python3 oneapi_tensorflow_example.py

echo "oneAPI 和 TensorFlow 安装及示例运行完成。"
```

### 脚本说明：
1. **配置Mesa 3D图形库**：通过安装Mesa库为OpenGL和Vulkan提供支持，并通过`glmark2`检查显卡的运行情况。
2. **安装Intel GPU依赖**：下载并安装Intel GPU的相关依赖，包括OpenCL和Level Zero等必需包。
3. **安装oneAPI库**：配置Intel的APT仓库，并安装`oneAPI`相关的运行时库（如DPC++和MKL）。
4. **安装oneAPI BaseKit**：下载并安装Intel的oneAPI BaseKit。
5. **安装TensorFlow和扩展**：安装TensorFlow，并添加Intel提供的TensorFlow GPU扩展以利用oneAPI运行时。
6. **检查配置**：使用Intel的`env_check.sh`脚本检查环境配置是否正确。
7. **运行示例代码**：运行一个简单的TensorFlow卷积操作以验证安装是否成功。

该脚本会自动执行所有步骤，确保环境配置正确，并在Intel平台上成功运行oneAPI和TensorFlow扩展。
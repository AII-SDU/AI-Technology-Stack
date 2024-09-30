# OpenXLA (AMD)

以下是一个在AMD平台上安装OpenXLA的自动化脚本。该脚本假设您已经安装了ROCm（用于AMD GPU的并行计算平台），并配置好了ROCm相关的库和工具。

```bash
#!/bin/bash

# 步骤 1: 更新系统并安装基本依赖
echo "更新系统并安装必要依赖..."
sudo apt-get update
sudo apt-get install -y build-essential git cmake python3 python3-pip python3-venv

# 步骤 2: 安装 ROCm (如果还未安装)
# 注意：此步骤假设您使用的是 Ubuntu 20.04/22.04
echo "安装 ROCm..."
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.5/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-dkms rocm-dev rocm-libs hipcub rocprim

# 步骤 3: 设置 ROCm 环境变量
echo "设置 ROCm 环境变量..."
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 步骤 4: 验证 ROCm 安装
echo "验证 ROCm 安装..."
/opt/rocm/bin/rocminfo
/opt/rocm/opencl/bin/x86_64/clinfo

# 步骤 5: 克隆 OpenXLA 项目
echo "克隆 OpenXLA 项目..."
git clone https://github.com/openxla/openxla.git
cd openxla

# 步骤 6: 创建 Python 虚拟环境
echo "创建 Python 虚拟环境..."
python3 -m venv venv
source venv/bin/activate

# 步骤 7: 安装 OpenXLA 依赖
echo "安装 OpenXLA 依赖..."
pip install --upgrade pip
pip install -r requirements.txt

# 步骤 8: 安装 Bazel (用于构建 OpenXLA)
echo "安装 Bazel..."
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings/bazel-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt-get update && sudo apt-get install bazel

# 步骤 9: 编译 OpenXLA
echo "编译 OpenXLA..."
bazel build ...

# 步骤 10: 验证 OpenXLA 安装
echo "验证 OpenXLA 安装..."
python -c "import openxla; print('OpenXLA 安装成功！')"

echo "OpenXLA 安装完成！"
```

### 脚本说明：
1. **安装依赖**：首先更新系统，并安装必要的依赖，包括构建工具和Python包管理工具。
2. **安装ROCm**：如果ROCm未安装，使用这个步骤来安装ROCm，包括`rocm-dkms`、`rocm-dev`等基本工具。ROCm是AMD平台上的并行计算框架，类似于NVIDIA的CUDA。
3. **设置ROCm环境变量**：配置系统的环境变量，确保ROCm工具可以在终端中正确使用。
4. **验证ROCm安装**：通过运行`rocminfo`和`clinfo`验证ROCm是否正确安装。
5. **克隆OpenXLA项目**：从GitHub克隆OpenXLA的源代码。
6. **创建Python虚拟环境**：使用Python虚拟环境隔离依赖，确保项目运行环境独立。
7. **安装OpenXLA依赖**：通过`pip`安装OpenXLA所需的Python依赖项。
8. **安装Bazel**：Bazel是OpenXLA的构建工具，这一步骤安装Bazel，确保可以正确编译OpenXLA。
9. **编译OpenXLA**：使用Bazel编译OpenXLA源代码。
10. **验证OpenXLA安装**：通过Python导入OpenXLA模块，检查是否安装成功。
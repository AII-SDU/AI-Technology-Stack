# ROCm / HIP

```bash
#!/bin/bash

# 步骤 1: 添加 ROCm 存储库
echo "添加 ROCm 存储库..."
# 添加 GPG 密钥
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# 添加存储库地址到 sources.list.d
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 步骤 2: 更新 APT 包缓存
echo "更新 APT 包缓存..."
sudo apt update

# 步骤 3: 安装 ROCm 软件包
echo "安装 ROCm 堆栈..."
sudo apt install -y rocm-dkms

# 步骤 4: 设置环境变量
echo "设置 ROCm 环境变量..."
echo 'export PATH=$PATH:/opt/rocm/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib' >> ~/.bashrc

# 使环境变量生效
echo "应用环境变量..."
source ~/.bashrc

# 步骤 5: 验证安装
echo "验证 ROCm 安装..."
/opt/rocm/bin/rocminfo

# 步骤 6: 安装 HIP 编程环境
echo "安装 HIP 基础工具..."
sudo apt install -y hip-base

# 步骤 7: 验证编译示例程序
echo "克隆并编译 HIP 示例程序..."
git clone https://github.com/ROCm-Developer-Tools/HIP-Examples.git
cd HIP-Examples/vectorAdd
make

# 验证成功运行
echo "运行 vectorAdd 示例程序..."
./vectorAdd

echo "ROCm 和 HIP 安装及测试完成。"
```

### 脚本说明：
1. **添加ROCm存储库**：从AMD官方源下载GPG密钥并添加软件包存储库。
2. **更新APT包缓存**：更新系统的包缓存。
3. **安装ROCm堆栈**：通过APT安装整个ROCm堆栈，包含核心驱动、工具和库。
4. **设置环境变量**：将ROCm路径添加到环境变量`PATH`和`LD_LIBRARY_PATH`中，并立即生效。
5. **验证安装**：使用`rocminfo`命令验证ROCm是否正确安装。
6. **安装HIP编程环境**：安装HIP的基本工具和SDK。
7. **编译并运行HIP示例程序**：克隆AMD提供的HIP示例代码库，编译并运行`vectorAdd`示例，验证HIP编程环境是否成功搭建。

该脚本会自动执行所有步骤，确保ROCm和HIP在AMD平台上的正确安装和配置。


## 算子库


以下是您提供的ROCm相关算子库的编译过程总结的自动化脚本，涵盖了`hipBLAS-common`、`hipBLASLt`、`rocBLAS`等库的安装和自定义编译步骤。

```bash
#!/bin/bash

# 步骤 1: 克隆并编译 hipBLAS-common
echo "克隆并编译 hipBLAS-common..."
git clone https://github.com/ROCm/hipBLAS-common.git
cd hipBLAS-common
mkdir build && cd build
cmake ..
make package install
cd ../..

# 步骤 2: 安装 hipBLASLt
echo "安装 hipBLASLt 开发包..."
sudo apt-get install -y hipblas-dev hipblaslt-dev

# 运行 hipBLASLt 安装脚本
echo "运行 hipBLASLt 安装脚本..."
git clone https://github.com/ROCm/hipBLASLt.git
cd hipBLASLt
./install.sh -idc --legacy_hipblas_direct --architecture 'gfx1100'
cd ..

# 步骤 3: 安装 rocBLAS 并自定义编译
echo "安装 rocBLAS..."
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS
./install.sh -idc

# 自定义编译 rocBLAS 测试程序
echo "自定义编译 rocBLAS 测试程序..."
sudo hipcc -o rocblas_test clients/samples/rocblas_test.cpp -lrocblas
cd ..

# 步骤 4: 编译 hipBLAS 测试程序
echo "编译 hipBLAS 测试程序..."
sudo hipcc -o hipblas_test hipBLAS-common/clients/samples/hipblas_test.cpp -I/opt/rocm/include/hipblas -lhipblas

# 步骤 5: 设置 ROCm 环境变量
echo "设置 ROCm 环境变量..."
export HIP_PATH=/opt/rocm
export PATH=$PATH:$HIP_PATH/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HIP_PATH/lib

echo "ROCm 算子库编译完成。"
```

### 脚本说明：
1. **hipBLAS-common**：克隆并编译`hipBLAS-common`库，创建`build`目录，使用CMake进行配置和编译。
2. **hipBLASLt**：安装`hipBLAS`和`hipBLASLt`开发包，运行`hipBLASLt`的安装脚本，并指定相关架构。
3. **rocBLAS**：克隆并安装`rocBLAS`，并编译自定义的`rocblas_test`测试程序。
4. **hipBLAS测试**：使用`hipcc`编译`hipBLAS`的测试程序，并包含必要的头文件和库。
5. **设置环境变量**：将`HIP_PATH`和`LD_LIBRARY_PATH`设置为ROCm的安装路径，确保运行时可以找到必要的工具和库。

此脚本将自动执行ROCm相关算子库的安装和编译步骤，确保环境正确配置和可执行程序生成。
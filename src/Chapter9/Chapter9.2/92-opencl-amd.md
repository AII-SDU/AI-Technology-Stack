# OpenCL (AMD)

```bash
#!/bin/bash

# 更新系统软件包
echo "更新系统软件包..."
sudo apt update && sudo apt upgrade -y

# 添加 ROCm 官方软件源
echo "添加 ROCm 官方软件源..."
wget -qO - http://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新软件包索引
echo "更新软件包索引..."
sudo apt update

# 安装 ROCm 和 OpenCL Runtime
echo "安装 ROCm 和 OpenCL Runtime..."
sudo apt install -y rocm-dkms rocm-opencl rocm-opencl-dev

# 设置环境变量
echo "设置环境变量..."
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# 使环境变量生效
echo "应用环境变量..."
source ~/.bashrc

# 验证 OpenCL 安装
echo "验证 OpenCL 安装..."
clinfo

# 安装必要的依赖库（可选）
echo "安装必要的依赖库..."
sudo apt install -y ocl-icd-libopencl1 opencl-headers clinfo

echo "OpenCL 安装完成。"
```

### 脚本说明：
1. 脚本会首先更新系统的软件包，然后添加ROCm仓库。
2. 安装ROCm驱动和OpenCL Runtime。
3. 自动将环境变量添加到`~/.bashrc`文件，并立即生效。
4. 通过`clinfo`验证OpenCL安装是否成功。
5. 可选步骤是安装一些OpenCL开发所需的依赖库。 

运行该脚本前，建议先确保操作系统版本支持ROCm，并根据需求修改部分配置。
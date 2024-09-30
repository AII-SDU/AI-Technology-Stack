# TPU MLIR


```bash
#!/bin/bash

# 步骤 1: 安装依赖工具 (p7zip、Docker)
echo "安装必要的依赖工具..."
sudo apt-get update
sudo apt-get install -y p7zip p7zip-full docker.io dkms libncurses5 gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

# 步骤 2: 解压SDK压缩包
echo "解压SDK压缩包..."
SDK_FILE="Release_<date>-public.zip"
7z x $SDK_FILE
cd Release_<date>-public

# 步骤 3: 配置Docker
echo "配置Docker..."
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
sudo service docker restart

# 步骤 4: 初始化tpu-mlir环境
echo "初始化tpu-mlir环境..."
cd tpu-mlir_<date>_<hash>
mkdir tpu-mlir
tar zxvf tpu-mlir_v<x.y.z>-<hash>-<date>.tar.gz --strip-components=1 -C tpu-mlir
cd tpu-mlir
docker run --privileged --name tpu_mlir_container -v $PWD:/workspace -it sophgo/tpuc_dev:v2.2
cd /workspace/tpu-mlir
source ./envsetup.sh

# 步骤 5: 初始化tpu-nntc环境
echo "初始化tpu-nntc环境..."
cd ../tpu-nntc_<date>_<hash>
mkdir tpu-nntc
tar zxvf tpu-nntc_v<x.y.z>-<hash>-<date>.tar.gz --strip-components=1 -C tpu-nntc
cd tpu-nntc
docker run -v $PWD/..:/workspace -p 8001:8001 -it sophgo/tpuc_dev:v2.1
cd /workspace/tpu-nntc
source scripts/envsetup.sh

# 步骤 6: 安装libsophon驱动和依赖库
echo "安装libsophon..."
cd ../libsophon_<date>_<hash>
sudo apt install -y dkms libncurses5
sudo dpkg -i sophon-*.deb
source /etc/profile

# 验证驱动安装
if ls /dev/bm* | grep -q 'bm-sophon'; then
    echo "libsophon 驱动安装成功！"
else
    echo "libsophon 驱动安装失败，请检查！"
fi

# 步骤 7: 安装sophon-mw
echo "安装sophon-mw..."
cd ../sophon-mw_<date>_<hash>
sudo dpkg -i sophon-mw-sophon-ffmpeg_*.deb sophon-mw-sophon-ffmpeg-dev_*.deb
sudo dpkg -i sophon-mw-sophon-opencv_*.deb sophon-mw-sophon-opencv-dev_*.deb
source /etc/profile

# 步骤 8: 设置交叉编译环境
echo "设置交叉编译环境..."
cd ../sophon-img_<date>_<hash>
mkdir -p soc-sdk
tar -zxf libsophon_soc_<x.y.z>_aarch64.tar.gz
cp -rf libsophon_soc_<x.y.z>_aarch64/opt/sophon/libsophon-<x.y.z>/lib soc-sdk
cp -rf libsophon_soc_<x.y.z>_aarch64/opt/sophon/libsophon-<x.y.z>/include soc-sdk
tar -zxf sophon-mw-soc_<x.y.z>_aarch64.tar.gz
cp -rf sophon-mw-soc_<x.y.z>_aarch64/opt/sophon/sophon-ffmpeg_<x.y.z>/lib soc-sdk
cp -rf sophon-mw-soc_<x.y.z>_aarch64/opt/sophon/sophon-ffmpeg_<x.y.z>/include soc-sdk
cp -rf sophon-mw-soc_<x.y.z>_aarch64/opt/sophon/sophon-opencv_<x.y.z>/lib soc-sdk
cp -rf sophon-mw-soc_<x.y.z>_aarch64/opt/sophon/sophon-opencv_<x.y.z>/include soc-sdk

# 步骤 9: 验证交叉编译环境
echo "验证交叉编译环境..."
if which aarch64-linux-gnu-g++; then
    echo "交叉编译环境设置成功！"
else
    echo "交叉编译环境设置失败！"
fi

echo "Sophon SDK 安装及环境配置完成。"
```

### 脚本说明：
1. **安装依赖工具**：安装必要的工具包，包括`p7zip`用于解压SDK包，`docker.io`用于Docker环境配置，`dkms`和`libncurses5`等库作为依赖。
2. **解压SDK**：解压SDK的压缩包，并进入解压后的目录。
3. **Docker配置**：安装并配置Docker，确保可以在容器中运行tpu-mlir和tpu-nntc。
4. **tpu-mlir和tpu-nntc环境初始化**：分别初始化tpu-mlir和tpu-nntc环境，在容器内运行必要的脚本。
5. **libsophon安装**：安装libsophon驱动和依赖库，确保开发环境和运行环境中的设备可以正常使用。
6. **sophon-mw安装**：安装sophon-mw相关工具，确保开发环境中已安装必要的多媒体支持库（FFmpeg和OpenCV）。
7. **交叉编译环境配置**：解压和配置用于交叉编译的库和头文件，并验证交叉编译工具链的设置是否成功。

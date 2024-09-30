# MUSA

```bash
#!/bin/bash

# 步骤 1: 检查系统环境是否正常
echo "检查系统环境..."
mthreads-gmi | grep "Driver Version:2.7.0"
if [ $? -ne 0 ]; then
    echo "系统环境不正常，请确保已正确安装MUSA驱动程序及相关组件。"
    exit 1
else
    echo "系统环境正常，继续执行..."
fi

# 步骤 2: 获取Docker镜像
echo "拉取Torch MUSA开发镜像..."
# 根据需要更换不同的Python版本，例如 py38/py39
sudo docker pull registry.mthreads.com/mcconline/musa-pytorch-dev-public:rc3.0.1-v1.2.1-S4000-py310
sudo docker pull registry.mthreads.com/mcconline/musa-pytorch-dev-public:rc3.0.1-v1.2.1-S3000-py310
sudo docker pull registry.mthreads.com/mcconline/musa-pytorch-dev-public:rc3.0.1-v1.2.1-S80-py310

# 步骤 3: 运行Docker镜像
echo "运行Torch MUSA开发环境镜像..."
sudo docker run -it \
    --privileged \
    --name=torch_musa_dev \
    --env MTHREADS_VISIBLE_DEVICES=all \
    registry.mthreads.com/mcconline/musa-pytorch-dev-public:rc3.0.1-v1.2.1-S80-py310 \
    /bin/bash

# 步骤 4: 使用容器中的工具和检查torch_musa
echo "检查Torch MUSA是否正常运行..."
cd torch_musa
bash scripts/run_unittest.sh

# 如果对torch_musa进行了代码修改，可以通过以下命令进行编译和安装
# echo "编译并安装新的torch_musa..."
# bash build.sh -c

echo "Torch MUSA 开发环境安装及检查完成。"
```

### 脚本说明：
1. **检查系统环境**：使用`mthreads-gmi`命令检查MUSA驱动版本是否为2.7.0，确保系统环境正常。
2. **获取Docker镜像**：拉取预构建的MUSA开发镜像，镜像中包含了PyTorch和torch_musa项目的运行环境。
3. **运行Docker容器**：运行Docker容器，将所有可见的设备暴露给容器，进入镜像内的Bash环境。
4. **检查torch_musa运行情况**：进入容器中的`torch_musa`目录，执行单元测试脚本检查`torch_musa`是否正常运行。
5. **编译并安装修改后的torch_musa**（可选）：如果修改了`torch_musa`代码，可以通过执行`build.sh`脚本重新编译和安装。


# ONNX (AMD)

```bash
#!/bin/bash

# 步骤 1: 验证是否已安装 Radeon Software for Linux (包含 ROCm)
echo "检查是否已安装 ROCm..."
if ! dpkg -l | grep -i rocm; then
    echo "未检测到 ROCm，确保您已成功安装 Radeon Software for Linux (带有 ROCm)。"
    exit 1
else
    echo "ROCm 已安装。"
fi

# 步骤 2: 验证是否已安装 MIGraphX
echo "检查 MIGraphX 安装情况..."
dpkg -l | grep migraphx
if [ $? -ne 0 ]; then
    echo "MIGraphX 未安装，请先安装 MIGraphX。"
    exit 1
else
    echo "MIGraphX 已安装。"
fi

# 验证是否已安装 half 库
echo "检查 half 库安装情况..."
dpkg -l | grep half
if [ $? -ne 0 ]; then
    echo "half 库未安装，正在安装 half 库..."
    sudo apt install half -y
else
    echo "half 库已安装。"
fi

# 步骤 3: 使用 PIP 安装 ONNX Runtime
echo "卸载现有的 onnxruntime-rocm 和 numpy..."
pip3 uninstall onnxruntime-rocm numpy -y

echo "安装 ONNX Runtime 和 numpy..."
pip3 install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.1.3/onnxruntime_rocm-1.17.0-cp310-cp310-linux_x86_64.whl numpy==1.26.4

# 步骤 4: 验证 ONNX Runtime 安装
echo "验证 ONNX Runtime 安装..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print('可用的执行提供者:', providers)
if 'MIGraphXExecutionProvider' in providers and 'ROCMExecutionProvider' in providers and 'CPUExecutionProvider' in providers:
    print('ONNX Runtime 安装成功！')
else:
    print('ONNX Runtime 安装失败，请检查日志。')
"

echo "ONNX Runtime 安装完成。"
```

### 脚本说明：
1. **验证ROCm和MIGraphX安装**：首先检查是否安装了Radeon Software for Linux和MIGraphX。如果没有安装MIGraphX，则会提示错误。
2. **验证`half`库安装**：检查是否已安装`half`库，如果未安装，则自动安装它。
3. **安装ONNX Runtime**：通过PIP安装预构建的ONNX Runtime wheel文件，并将`numpy`降级到兼容的版本`1.26.4`。
4. **验证ONNX Runtime安装**：通过Python脚本检查ONNX Runtime的可用执行提供者（Execution Providers），确认MIGraphX、ROCM和CPU执行提供者是否可用。

运行此脚本将自动完成AMD平台上ONNX Runtime的安装和配置。
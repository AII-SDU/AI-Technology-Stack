# SYCL (AMD)

```bash
#!/bin/bash

# 步骤 1: 安装必要依赖
echo "安装必要依赖..."
sudo apt -y install cmake pkg-config build-essential

# 步骤 2: 下载并安装英特尔 oneAPI 工具包
echo "下载并安装 Intel oneAPI BaseKit..."
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100_offline.sh
sudo sh ./l_BaseKit_p_2024.2.1.100_offline.sh -a --silent --cli --eula accept

# 步骤 3: 下载并安装 AMD 对应 DPC++/C++ 插件
echo "下载并安装 AMD 对应的 DPC++/C++ 插件..."
wget https://developer.codeplay.com/products/oneapi/amd/download/oneapi-for-amd-gpus-2024.2.1-rocm-5.4.3-linux.sh
sudo sh oneapi-for-amd-gpus-2024.2.1-rocm-5.4.3-linux.sh

# 步骤 4: 设置环境变量
echo "设置 oneAPI 环境变量..."
source /opt/intel/oneapi/setvars.sh --include-intel-llvm

# 步骤 5: 验证 SYCL 安装
echo "验证 SYCL 安装..."
sycl-ls

# 步骤 6: 创建 SYCL 示例程序
echo "创建 SYCL 示例程序..."
cat <<EOL > simple-sycl-app.cpp
#include <sycl/sycl.hpp>

int main() {
  sycl::buffer<int, 1> Buffer{4};
  sycl::queue Queue{};
  sycl::range<1> NumOfWorkItems{Buffer.size()};

  Queue.submit([&](sycl::handler &cgh) {
    auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](sycl::id<1> WIid) {
          Accessor[WIid] = static_cast<int>(WIid.get(0));
        });
  });

  auto HostAccessor = Buffer.get_host_access();
  bool MismatchFound{false};
  for (size_t I{0}; I < Buffer.size(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}
EOL

# 步骤 7: 获取 AMD GPU 的架构名称
echo "获取 AMD GPU 架构信息..."
ARCH=$(rocminfo | grep 'Name: *gfx' | awk '{print $2}')

# 步骤 8: 编译 SYCL 示例程序
echo "编译 SYCL 示例程序..."
icpx -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=$ARCH -o simple-sycl-app simple-sycl-app.cpp

# 步骤 9: 运行 SYCL 示例程序
echo "运行 SYCL 示例程序..."
ONEAPI_DEVICE_SELECTOR="hip:*" SYCL_PI_TRACE=1 ./simple-sycl-app

echo "SYCL 安装和示例运行完成。"
```

### 脚本说明：
1. **安装依赖**：首先安装构建SYCL程序所需的基本工具如CMake、pkg-config等。
2. **下载并安装Intel oneAPI工具包**：下载并安装Intel提供的oneAPI BaseKit，用于支持DPC++编译器和SYCL环境。
3. **安装AMD对应的DPC++插件**：从Codeplay官网下载并安装针对AMD GPU的DPC++插件，确保AMD GPU可以使用SYCL编译和运行。
4. **设置环境变量**：使用Intel提供的`setvars.sh`脚本配置环境变量，确保工具链和库正确配置。
5. **验证SYCL安装**：通过`sycl-ls`命令验证SYCL平台和设备是否正确安装。
6. **创建并编译SYCL程序**：编写一个简单的SYCL程序来测试平台配置，并使用`icpx`编译该程序，针对AMD GPU架构生成代码。
7. **运行SYCL程序**：设置环境变量`ONEAPI_DEVICE_SELECTOR`为`hip:*`以选择AMD GPU，并运行编译好的程序进行测试。

该脚本将自动执行所有步骤，确保SYCL在AMD平台上的正确安装和运行，并且提供了一个简单的示例以验证环境配置。
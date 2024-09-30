# AI 技术栈解析及应用

## 目录

1. [引言](#1-引言)

2. [技术栈架构概述](#2-技术栈架构概述)

   2.1 [AI 技术栈层次](#21-ai-技术栈层次)

   2.2 [AI技术栈的的意义](#22-ai-技术栈的的意义)

   2.3 [AI技术栈初步应用](#23-ai-技术栈初步应用)

3. [NVIDIA 平台](#3-nvidia-平台)

   3.1 [CUDA](#31-cuda)

   3.2 [OpenCL (NVIDIA)](#32-opencl-nvidia)

   3.3 [SYCL (DPC++) (NVIDIA)](#33-sycl-dpc-nvidia)

   3.4 [Triton (NVIDIA)](#34-triton-nvidia)

   3.5 [Apache TVM (NVIDIA)](#35-apache-tvm-nvidia)

   3.6 [OpenXLA (NVIDIA)](#36-openxla-nvidia)

   3.7 [OpenACC](#37-openacc)

4. [AMD 平台](#4-amd-平台)

   4.1 [ROCm / HIP](#41-rocm--hip)

   4.2 [OpenCL (AMD)](#42-opencl-amd)

   4.3 [SYCL (AMD)](#43-sycl-amd)

   4.4 [Triton (AMD)](#44-triton-amd)

   4.5 [Apache TVM (AMD)](#45-apache-tvm-amd)

   4.6 [OpenXLA (AMD)](#46-openxla-amd)

   4.7 [ONNX (AMD)](#47-onnx-amd)
5. [Intel 平台](#5-intel-平台)

   5.1 [oneAPI](#51-oneapi)

6. [算能 TPU 平台](#6-算能-tpu-平台)

   6.1 [TPU MLIR](#61-tpu-mlir)

7. [摩尔线程平台](#7-摩尔线程平台)

   7.1 [MUSA](#71-bang-c)

8. [总结与展望](#8-总结与展望)

9. [附录——AI技术栈安装指南 ](#9-AI技术栈安装指南)

## 1. 引言

人工智能（AI）技术在过去十年中经历了前所未有的发展，从学术研究走向广泛的商业应用。这一快速发展不仅带来了令人瞩目的技术突破，也催生了复杂多样的软硬件生态系统。在这个快速演进的领域中，理解和掌握AI技术栈的结构和特点变得越来越重要。

本文旨在提供一个系统化的视角来审视当前AI技术栈的构成。我们将深入探讨从底层硬件到高层应用框架的各个层次，分析它们之间的相互关系，以及如何协同工作以支持现代AI系统的开发和部署。通过这种分层的方法，我们不仅可以更好地理解现有技术，还能洞察未来的发展趋势。

在接下来的章节中，我们将首先概述AI技术栈的整体架构，然后逐一深入探讨各大主流平台（如NVIDIA、AMD、Intel等）的技术特点。我们将分析每个平台在各个层次的实现，比较它们的优势和局限性，并探讨如何在实际应用中做出最优的技术选择。

通过本文，我们希望为AI研究者、开发者和决策者提供一个全面的参考框架，帮助他们在这个快速发展的领域中做出明智的技术决策，并为未来的创新铺平道路。

## 2. 技术栈架构概述

### 2.1 AI 技术栈层次

人工智能技术的快速发展带来了复杂多样的软硬件生态系统。为了更好地理解和利用这些技术，我们提出了一种新的分层方法来分析AI技术栈。这种分层不仅有助于我们理清各种技术之间的关系，还为开发者、研究者和决策者提供了一个系统化的视角来审视整个AI生态系统。

AI 技术栈通常包含以下层次：

1. **系统软件层**：设备驱动程序、底层 API
2. **运行时环境层**：执行环境和运行时库
3. **编程模型和语言层**：特定于硬件的编程语言和模型
4. **计算库层**：优化的数学和深度学习库
5. **框架模型层**：高级深度学习框架

系统软件层是整个技术栈的基础，它直接与硬件交互，提供底层的驱动程序和API。这一层的设计和优化直接影响了整个系统的性能和稳定性。运行时环境层则在系统软件之上提供了一个抽象层，使得上层应用能够更加高效地利用硬件资源。

编程模型和语言层是开发者与系统交互的主要接口。不同的编程模型和语言反映了不同的计算范式和抽象级别，从而影响了开发效率和代码可移植性。计算库层提供了高度优化的数学和机器学习算法实现，是提升性能的关键所在。最上层的框架模型层则为开发者提供了高级的API和工具，大大简化了AI模型的开发和部署过程。


### 2.2 AI 技术栈的的意义

AI技术栈的分层方法不仅仅是一种理论构造，它在实际应用和研究中具有深远的意义和多方面的优势：

1. **系统化理解**：分层结构提供了一个系统化的框架，使得复杂的AI生态系统变得更加清晰可理解。这种结构化的视角有助于开发者、研究者和决策者更好地把握整个技术领域的全貌。

2. **模块化设计**：分层架构促进了模块化设计的思想。每一层都有明确定义的接口和功能，这使得开发者可以专注于特定层次的优化，而不必过多考虑其他层次的复杂性。

3. **技术对比**：通过分层，我们可以在相同的层次上比较不同平台或技术的实现。这种横向对比有助于识别各种技术的优势和劣势，为技术选型提供客观依据。

4. **性能优化**：分层结构使得性能瓶颈的定位变得更加精确。开发者可以针对特定层次进行优化，而不是盲目地对整个系统进行调整。

5. **跨层优化**：虽然分层提供了清晰的结构，但它也为跨层优化提供了可能。了解各层之间的相互作用，可以实现更深层次的系统优化。

6. **标准化促进**：分层架构为制定行业标准提供了基础。不同层次的标准化有助于提高技术的互操作性和可移植性。

总的来说AI技术栈提供了一个清晰的结构来理解和比较不同的AI技术。例如，当我们比较NVIDIA的CUDA和AMD的ROCm时，我们可以在每一层级进行对比，从而全面地评估两种技术的异同。这不仅有助于技术选型，还为性能优化提供了指导。

从开发者的角度来看，这种分层结构使得他们可以根据自己的需求和专长选择合适的切入点。例如，深度学习研究者可能主要关注框架模型层，而系统优化专家则可能更多地工作在底层。同时，这种分层也有利于跨层优化，开发者可以根据需要在不同层次间进行调优。

从行业发展的角度来看，这种分层结构也反映了AI技术的发展趋势。我们看到，在每一层都有不断涌现的新技术，如编程模型层的SYCL，计算库层的oneDNN，以及框架模型层的各种新兴深度学习框架。这种分层结构有助于我们更好地理解这些新技术在整个生态系统中的位置和作用。

通过这种分层方法，我们不仅能更好地理解和利用现有技术，还能为未来的技术发展提供清晰的路径和方向。

### 2.3 AI 技术栈初步应用

AI 技术栈的每个层次分析都有其特定方法和应用demo。本节将阐述后续章节的分析逻辑，解释为什么要进行这样的分层分析，以及每层分析的意义和应用。通过深入理解每个层次的特点，我们可以更好地利用 AI 技术栈来开发和优化 AI 系统。

#### 2.3.1 系统软件层和运行时环境层

在后续章节中，这一层的分析主要聚焦于 API 调用和硬件接口，目的是理解不同技术路线在相同硬件平台下如何与底层系统交互。

1. **API 调用分析**
   - 目的：了解各种 AI 框架和库如何与底层硬件交互
   - 意义：揭示谁实际使用了 CUDA Driver API，CUDA Runtime API等底层接口，有助于理解不同技术路线调用相同接口的异同。

2. **硬件接口比较**
   - 目的：比较不同 AI 技术栈在访问相同硬件时的方式
   - 意义：了解不同方案的底层实现差异，为性能优化提供思路

3. **扩展性分析**
   - 目的：研究如何为新硬件或新接口扩展现有系统
   - 意义：为未来硬件适配和系统升级提供指导

这一层不进行直接的性能比较，因为系统软件层的差异通常不是性能瓶颈的主要来源。相反，我们关注的是不同方案如何利用底层资源，这为理解整体性能提供了基础。

#### 2.3.2 编程模型和语言层

这一层的分析主要起到教学和概念引入的作用，为后续的深入分析奠定基础。

1. **语言特性对比**
   - 目的：展示不同编程语言（如 Python、C++、CUDA）在 AI 开发中的应用
   - 意义：帮助理解语言选择对开发效率和性能的影响

2. **算子编写示例**
   - 目的：提供常见 AI 算子（如卷积、矩阵乘法）的实现示例
   - 意义：深入理解算子工作原理，为后续优化提供思路

3. **并行计算模型介绍**
   - 目的：解释 CUDA、OpenCL 等并行计算模型的基本概念
   - 意义：为理解 GPU 加速原理和优化方法打下基础

这一层的分析不直接进行性能比较，而是为读者提供必要的背景知识，使他们能够理解后续章节中更复杂的性能分析和优化策略。

#### 2.3.3 计算库层、框架模型层和模型层

在这些高层次中，我们将基于现有的 [AI Benchmark](https://github.com/AII-SDU/AI-Benchmark-SDU)进行更深入的应用和研究。

1. **计算库性能分析**
   - 目的：比较不同计算库（如 cuDNN、oneDNN）在常见算子上的性能
   - 意义：了解底层库对整体性能的影响，指导算子优化和选择

2. **框架性能对比**
   - 目的：评估不同深度学习框架（如 TensorFlow、PyTorch）在相同任务上的性能
   - 意义：帮助开发者选择适合特定任务的框架，了解框架优化的重要性

3. **模型层 Benchmark 扩展**
   - 目的：将更多类型的模型纳入 AI Benchmark
   - 意义：提供更全面的性能评估，覆盖更广泛的应用场景

4. **算子级 Benchmark**
   - 目的：开发针对单个算子的性能测试套件
   - 意义：深入了解性能瓶颈，指导底层优化

5. **安装和部署指南**
   - 目的：基于 Benchmark 结果，提供模型选择和部署的最佳实践
   - 意义：帮助用户根据自身硬件和需求选择最合适的模型和框架

这些高层次的分析直接关系到 AI 系统的最终性能。通过全面的 Benchmark 和分析，我们可以获得不同组件和配置的详细性能数据，从而指导实际应用中的选择和优化。

通过这种分层分析方法，我们可以全面地理解 AI 技术栈的各个层次，从底层硬件接口到高层模型性能。这种方法不仅有助于理解当前 AI 系统的性能特征，还为未来的优化和创新提供了清晰的路径。在后续章节中，我们将基于这个框架，提供具体的示例和深入分析，展示如何在实际应用中利用这种分层思想来优化 AI 系统性能。

## 3. NVIDIA 平台


### 3.1 CUDA

#### 3.1.1 技术栈架构

CUDA（Compute Unified Device Architecture）是 NVIDIA 开发的并行计算平台和编程模型。CUDA 技术栈涵盖了从底层硬件到高层应用框架的多个层次，使开发者能够充分利用 NVIDIA GPU 的强大计算能力。以下是 CUDA 技术路线的主要组成部分

1. **系统软件层**
   - NVIDIA GPU 驱动：为 GPU 提供基本的系统级支持
   - CUDA Driver API：低级 API，提供对 GPU 的直接控制
     - 允许直接管理设备、内存分配和程序执行
     - 适用于需要细粒度控制的高级应用
     - 提供与 NVIDIA GPU 硬件交互的底层接口

2. **运行时环境层**
   - CUDA Runtime API：高级 API，简化了 GPU 编程，自动管理许多底层细节
     - 提供更高级的抽象，简化了 GPU 的使用
     - 自动处理上下文管理和程序加载等任务
     - 更适合一般开发者使用，提供了更好的易用性

3. **编程模型和语言层**
   - CUDA C/C++：扩展了 C/C++ 语言，允许开发者编写在 GPU 上运行的并行程序
     - 允许在 CPU 和 GPU 上混合编程
     - 使用 CUDA 特定语法（如 `__global__`）来定义 GPU 函数
     - 通过 `<<<>>>` 语法启动内核
     - 支持主机代码和设备代码的混合编写

4. **计算库层**
   - cuBLAS：用于线性代数计算的库
     - 提供 GPU 加速的矩阵运算和 BLAS 功能
     - 广泛用于深度学习中的矩阵计算
   - NCCL：用于多 GPU 通信的库
     - 支持多 GPU 之间的高效通信和数据交换
     - 主要用于分布式深度学习训练
   - 其他专用算子库（如 cuDNN）

5. **框架模型层**
   - PyTorch：支持动态计算图的深度学习框架
     - 通过 `torch.cuda` 模块提供 CUDA 功能
     - 自动管理 GPU 内存
     - 支持 CPU 和 GPU 之间的数据转移
   - TensorFlow：支持静态和动态计算图的深度学习框架
     - 通过 XLA 编译器优化 GPU 代码执行
     - 提供高级 API，简化了 CUDA API 的使用

##### 关系解析
![alt text](image.png)

1. **CUDA Driver API 和 CUDA Runtime API 的关系**
    - Runtime API 构建在 Driver API 之上，提供了更高级的抽象
    - Driver API 提供更多控制，但使用更复杂
    - Runtime API 更容易上手，隐藏了 Driver API 的复杂性
    - 开发者可以根据需求选择使用 Runtime API 或直接使用 Driver API
2. **PyTorch 和 TensorFlow 与 CUDA 的关系**
    - 两者都基于 CUDA Runtime API 实现 GPU 加速
    - 提供了高级抽象，使开发者无需直接编写 CUDA 代码
    - 支持自动微分和 GPU 加速的深度学习模型训练
    - PyTorch 和 TensorFlow 都支持 CPU 和 GPU 训练
3. **cuBLAS 和 NCCL 与 CUDA 的关系**
    - 这些库是 CUDA 生态系统的重要组成部分
    - 它们利用 CUDA 的并行计算能力，提供高性能的数学运算和通信功能
    - 与 CUDA C/C++ 和 CUDA API 结合使用，提供高性能计算能力

通过以上结构，CUDA 技术路线为开发者提供了从底层硬件控制到高层应用开发的全面支持，使得 GPU 并行计算的强大功能能够被有效地应用到各种计算密集型任务中。


#### 3.1.2 系统软件层

编写了一个使用 CUDA Driver API 的程序，列出系统中可用的 CUDA 设备，获取设备的名称、计算能力、驱动版本和全局内存大小，并创建和销毁 CUDA 上下文。

- 初始化 CUDA 驱动
- 获取可用 CUDA 设备的数量，并循环遍历每个设备
- 使用 cuDeviceGetName、cuDeviceGetAttribute 、cuDeviceTotalMem和cuDriverGetVersion 获取设备的详细信息
- 创建 CUDA 上下文并设置为当前上下文
- 输出设备信息，并在结束时销毁上下文

   示例代码：

```c++
#include <iostream>
#include <cuda.h>

// Check the return value of CUDA functions and print error message on failure
void checkCudaErrors(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr); 
        std::cerr << "CUDA Error: " << errorStr << std::endl;
        exit(EXIT_FAILURE); 
    }
}

// Print information about a CUDA device
void printDeviceInfo(CUdevice device) {
    int driverVersion = 0;
    char deviceName[256];
    // Get device name
    checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    
    int computeCapabilityMajor, computeCapabilityMinor;
    // Get the major and minor version of compute capability
    checkCudaErrors(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device)); 
    checkCudaErrors(cuDriverGetVersion(&driverVersion));

    // Print device details
    std::cout << "Device Name: " << deviceName << std::endl;
    std::cout << "Compute Capability: " << computeCapabilityMajor << "." << computeCapabilityMinor << std::endl;
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
    std::cout << "Total Global Memory: " << totalGlobalMem / (1024 * 1024) << " MB" << std::endl; 
}

int main() {
    // Initialize CUDA
    checkCudaErrors(cuInit(0)); 

    // Get the number of available CUDA devices
    int deviceCount;
    checkCudaErrors(cuDeviceGetCount(&deviceCount)); 
    std::cout << "Number of CUDA Devices: " << deviceCount << std::endl; 

    CUdevice device; 
    // Iterate through each device and print its information
    for (int i = 0; i < deviceCount; i++) {
        checkCudaErrors(cuDeviceGet(&device, i));
        printDeviceInfo(device);
        std::cout << std::endl;
    }

    CUcontext context;
    // Create a CUDA context and set it as the current context
    checkCudaErrors(cuCtxCreate(&context, 0, deviceCount > 0 ? device : 0)); 
    checkCudaErrors(cuCtxSetCurrent(context));

    std::cout << "CUDA context created successfully." << std::endl; 

    checkCudaErrors(cuCtxDestroy(context)); 

    return 0; 
}
```

结果：

```
Number of CUDA Devices: 1
Device Name: NVIDIA GeForce RTX 4080 SUPER
Compute Capability: 8.9
CUDA Driver Version: 12.4
Total Global Memory: 16072 MB

CUDA context created successfully.
```

#### 3.1.3 运行时环境层

CUDA Runtime API 是 NVIDIA 提供的用于管理和使用 GPU 资源的接口，旨在简化开发者与 CUDA 设备之间的交互。该 API 支持多种功能，包括设备查询、内存管理和流控制等，极大地提高了 GPU 编程的效率和可用性。

参考仓库地址：[deviceQuery](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/1_Utilities/deviceQuery/deviceQuery.cpp) 

示例代码如下：

```c++
/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);
  printf(
      " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "  Total amount of global memory:                 %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
#endif
    printf("%s", msg);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
    }

#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

#endif

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                : "WDDM (Windows Display Driver Model)");
#endif
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown", NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      // Only boards based on Fermi or later can support P2P
      if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && prop[i].tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                 prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                 can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  // csv masterlog info
  // *****************************
  // exe and CUDA driver name
  printf("\n");
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[16];

  // driver version
  sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000,
            (driverVersion % 100) / 10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
           (driverVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000,
            (runtimeVersion % 100) / 10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  // Device count
  sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d", deviceCount);
#else
  snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
  sProfileString += cTemp;
  sProfileString += "\n";
  printf("%s", sProfileString.c_str());

  printf("Result = PASS\n");

  // finish
  exit(EXIT_SUCCESS);
}
```

结果：

```
./Samples/1_Utilities/deviceQuery/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4080 SUPER"
  CUDA Driver Version / Runtime Version          12.4 / 12.3
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 16072 MBytes (16852844544 bytes)
  (080) Multiprocessors, (128) CUDA Cores/MP:    10240 CUDA Cores
  GPU Max Clock rate:                            2550 MHz (2.55 GHz)
  Memory Clock rate:                             11501 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 67108864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 12.3, NumDevs = 1
Result = PASS
```

#### 3.1.4 编程模型和语言层

[此处填写 CUDA 编程模型和语言层内容]

#### 3.1.5 计算库层

cuBLAS 是 NVIDIA 提供的高性能线性代数库，专为 CUDA 平台优化，支持多种基本线性代数操作，如矩阵乘法、向量运算和矩阵分解。cuBLAS 利用 GPU 的并行计算能力，提供高效的内存访问模式和自动优化的内核，能够显著提升矩阵运算的性能。

参考仓库地址：[cuda-samples](https://github.com/NVIDIA/cuda-samples) 

例如，矩阵乘法（GEMM）操作可以通过 cuBLAS 的简单接口实现。

示例代码如下：

```c++
// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// cuBLAS library
#include <cublas_v2.h>

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using cuBLAS
 */
int MatrixMultiply(int argc, char **argv,
                   const dim3 &dimsA,
                   const dim3 &dimsB) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 0.01f;
  ConstantInit(h_A, size_A, 1.0f);
  ConstantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C;
  checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Copy host memory to device
  checkCudaErrors(
      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(
      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));

  // Execute the cuBLAS matrix multiplication
  int nIter = 300;

  cublasHandle_t handle;
  cublasCreate(&handle);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (int j = 0; j < nIter; j++) {
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dimsB.x, dimsA.y, dimsA.x,
                &alpha,
                d_B, dimsB.x,
                d_A, dimsA.x,
                &beta,
                d_C, dimsB.x);
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf("cuBLAS Performance= %.2f GFlop/s, Time= %.3f msec\n",
         gigaFlops, msecPerMatrixMul);

  // Copy result from device to host
  checkCudaErrors(
      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  cublasDestroy(handle);
  // Clean up memory
  checkCudaErrors(cudaFreeHost(h_A));
  checkCudaErrors(cudaFreeHost(h_B));
  checkCudaErrors(cudaFreeHost(h_C));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  return EXIT_SUCCESS;
}


int main(int argc, char **argv) {
  printf("[Matrix Multiply Using cuBLAS] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  dim3 dimsA(320, 320, 1);
  dim3 dimsB(320, 320, 1);

  // Width of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "wA")) {
    dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
  }

  // Height of Matrix A
  if (checkCmdLineFlag(argc, (const char **)argv, "hA")) {
    dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
  }

  // Width of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "wB")) {
    dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
  }

  // Height of Matrix B
  if (checkCmdLineFlag(argc, (const char **)argv, "hB")) {
    dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  checkCudaErrors(cudaProfilerStart());
  int matrix_result = MatrixMultiply(argc, argv, dimsA, dimsB);
  checkCudaErrors(cudaProfilerStop());

  exit(matrix_result);
}
```

结果：

```
[Matrix Multiply Using cuBLAS] - Starting...
MatrixA(320,320), MatrixB(320,320)
cuBLAS Performance= 1752.85 GFlop/s, Time= 0.037 msec
```

#### 3.1.6框架模型层

[此处填写 CUDA 框架模型层内容]

#### 3.1.7 对比与思考

做了什么：
来源网址：
怎么做：
结果：

### 3.2 OpenCL (NVIDIA)

#### 3.2.1 技术栈架构

**1. 系统软件层**
- 设备驱动程序：
  - 为特定硬件（如 GPU、CPU、FPGA）提供底层支持
  - 实现 OpenCL 规范定义的功能
  - 处理设备特定的优化和功能
- OpenCL ICD (Installable Client Driver)：
  - 提供对多个 OpenCL 实现的支持
  - 允许在同一系统上共存多个 OpenCL 供应商的实现
  - 管理不同 OpenCL 实现之间的切换和交互

**2. 运行时环境层**
- OpenCL Runtime：
  - 提供 OpenCL API 的实现
  - 管理设备、上下文、命令队列和内存对象
  - 处理内核编译和执行
  - 协调主机和设备之间的数据传输
  - 支持事件和同步机制

**3. 编程模型和语言层**
- OpenCL C/C++：
  - 基于 C99 标准的编程语言，用于编写 OpenCL 内核
  - 支持向量数据类型和内置函数
  - 提供内存模型和同步原语
  - 允许编写可在各种设备上执行的并行代码
- OpenCL C++ 包装器：
  - 为 C++ 程序员提供面向对象的 API
  - 简化内存管理和错误处理
  - 提供更现代的 C++ 接口

**4. 计算库层**
- clBLAS：
  - OpenCL 实现的基本线性代数子程序（BLAS）库
  - 提供矩阵和向量操作的高性能实现
  - 支持多种设备类型
- clDNN (Compute Library for Deep Neural Networks)：
  - 用于深度学习的 OpenCL 加速库
  - 提供常见的神经网络层和操作
  - 优化for各种硬件平台

**5. 框架模型层**
- TensorFlow with OpenCL：
  - 通过 ComputeCpp 或其他 OpenCL 后端支持 OpenCL
  - 允许在支持 OpenCL 的设备上运行 TensorFlow 模型
- Caffe with OpenCL：
  - 使用 OpenCL 后端的 Caffe 深度学习框架
  - 支持在各种 OpenCL 设备上训练和推理
- OpenCV with OpenCL：
  - 计算机视觉库，集成了 OpenCL 支持
  - 利用 OpenCL 加速图像和视频处理操作
- ArrayFire：
  - 高性能计算库，支持 OpenCL 后端
  - 提供线性代数、信号处理和计算机视觉功能
  - 简化了 OpenCL 编程，提供高级抽象



![alt text](1271726043038_.pic.jpg)
##### 关系解析
OpenCL作为一个开放的异构计算框架，在模型层面支持硬件加速、跨设备兼容性和性能优化。它的核心组件包括OpenCL ICD、OpenCL Runtime和OpenCL C/C++语言。

OpenCL ICD (Installable Client Driver) 是一个关键组件，它允许多个OpenCL实现共存，提供了一个统一的接口来管理不同厂商的OpenCL实现。这种设计极大地增强了OpenCL的灵活性和可扩展性，使得开发者可以在不同的硬件平台上无缝切换。OpenCL Runtime负责管理设备、内存和任务调度等核心功能。它处理内存分配、数据传输、内核编译和执行等底层操作，为开发者提供了一个抽象层，简化了异构计算的复杂性。Runtime与ICD紧密协作，确保了OpenCL应用程序的高效运行。

在编程语言方面，OpenCL C/C++扩展了标准C/C++，增加了并行计算所需的特性。它支持向量数据类型、内存模型和并行编程构造，使得开发者能够充分利用异构计算资源。OpenCL 2.1引入了SPIR-V中间表示，进一步增强了跨平台兼容性和编译优化。clBLAS和clDNN是基于OpenCL的重要库，分别针对基础线性代数子程序和深度神经网络计算进行了优化。这些库充分利用了OpenCL的并行计算能力，为科学计算和机器学习应用提供了高性能解决方案。OpenCL与其他技术的集成也是其强大之处。例如，深度学习框架如PyTorch可以利用OpenCL进行GPU加速，而OpenCL本身也支持与CUDA等其他并行计算框架的互操作。

总的来说，OpenCL通过其灵活的架构、强大的运行时系统和丰富的编程接口，为异构计算提供了一个全面的解决方案。它不仅支持跨平台开发，还能够充分发挥各种计算设备的性能潜力，在高性能计算、图像处理、科学模拟等领域发挥着重要作用。OpenCL的生态系统持续发展，不断适应新的硬件架构和计算需求，为未来的并行计算和异构系统开发铺平了道路。
#### 3.2.2 系统软件层

该程序使用OpenCL API 列出了系统中所有可用的 NVIDIA 设备，包括设备名称、驱动版本、计算单元数量和全局内存大小，并创建和销毁了一个OpenCL上下文。

- **获取OpenCL平台**：使用`clGetPlatformIDs`获取系统中的所有 OpenCL 平台。

- **检查NVIDIA平台**：遍历平台列表，使用`clGetPlatformInfo`检查是否为 NVIDIA 平台。

- **获取设备信息**：通过`clGetDeviceIDs`获取 NVIDIA 平台中的所有设备，并使用`clGetDeviceInfo`获取每个设备的详细信息，如设备名称、驱动版本和全局内存大小。

- **创建和销毁上下文**：使用`clCreateContext`创建一个 OpenCL 上下文，并在使用后释放该上下文。

  示例代码：

```c++
#include <iostream>
#include <cuda.h>

// Check the return value of CUDA functions and print error message on failure
void checkCudaErrors(CUresult result) {
    if (result != CUDA_SUCCESS) {
        const char *errorStr;
        cuGetErrorString(result, &errorStr); 
        std::cerr << "CUDA Error: " << errorStr << std::endl;
        exit(EXIT_FAILURE); 
    }
}

// Print information about a CUDA device
void printDeviceInfo(CUdevice device) {
    int driverVersion = 0;
    char deviceName[256];
    // Get device name
    checkCudaErrors(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    
    int computeCapabilityMajor, computeCapabilityMinor;
    // Get the major and minor version of compute capability
    checkCudaErrors(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device)); 
    checkCudaErrors(cuDriverGetVersion(&driverVersion));

    // Print device details
    std::cout << "Device Name: " << deviceName << std::endl;
    std::cout << "Compute Capability: " << computeCapabilityMajor << "." << computeCapabilityMinor << std::endl;
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
    std::cout << "Total Global Memory: " << totalGlobalMem / (1024 * 1024) << " MB" << std::endl; 
}

int main() {
    // Initialize CUDA
    checkCudaErrors(cuInit(0)); 

    // Get the number of available CUDA devices
    int deviceCount;
    checkCudaErrors(cuDeviceGetCount(&deviceCount)); 
    std::cout << "Number of CUDA Devices: " << deviceCount << std::endl; 

    CUdevice device; 
    // Iterate through each device and print its information
    for (int i = 0; i < deviceCount; i++) {
        checkCudaErrors(cuDeviceGet(&device, i));
        printDeviceInfo(device);
        std::cout << std::endl;
    }

    CUcontext context;
    // Create a CUDA context and set it as the current context
    checkCudaErrors(cuCtxCreate(&context, 0, deviceCount > 0 ? device : 0)); 
    checkCudaErrors(cuCtxSetCurrent(context));

    std::cout << "CUDA context created successfully." << std::endl; 

    checkCudaErrors(cuCtxDestroy(context)); 

    return 0; 
}
```

结果：

```
Platform Name: NVIDIA CUDA
Device Name: NVIDIA GeForce RTX 4080 SUPER
Driver Version: 550.107.02
Max Compute Units: 80
Global Memory Size: 16072 MB

OpenCL context created successfully.
```

#### 3.2.3 运行时环境层

#### 3.2.4 编程模型和语言层

#### 3.2.5 计算库层

clBLAS 是一个开源的高性能线性代数库，专为 OpenCL 平台设计，支持多种基本线性代数操作，如矩阵乘法和矩阵-向量乘法。clBLAS 利用 OpenCL 的并行计算能力，提供灵活的内存管理和高效的内核优化，显著提升线性代数运算的性能。

参考仓库地址：[clBLAS](https://github.com/clMathLibraries/clBLAS) 

`clblasChemm` 展示了如何使用 clBLAS 进行复数矩阵的乘法操作。

示例代码如下：

```c++
/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#include <sys/types.h>
#include <stdio.h>
#include <string.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */
static const clblasOrder order = clblasRowMajor;

#define M  4
#define N  3

static const cl_float2 alpha = {{10, 10}};

static const clblasSide side = clblasLeft;
static const clblasUplo uplo = clblasLower;
static const cl_float2 A[M*M] = {
    {{11, 12}}, {{-1, -1}}, {{-1, -1}}, {{-1, -1}},
    {{21, 22}}, {{22, 23}}, {{-1, -1}}, {{-1, -1}},
    {{31, 32}}, {{32, 33}}, {{33, 34}}, {{-1, -1}},
    {{41, 61}}, {{42, 62}}, {{43, 73}}, {{44, 23}}
};
static const size_t lda = M;

static const cl_float2 B[M*N] = {
    {{11, -21}},  {{-12, 23}}, {{13, 33}},
    {{21, 12}},   {{22, -10}}, {{23, 5}},
    {{31, 1}},    {{-32, 65}}, {{33, -1}},
    {{1, 41}},    {{-33, 42}}, {{12, 43}},
};
static const size_t ldb = N;

static const cl_float2 beta = {{20, 20}};

static cl_float2 C[M*N] = {
    {{11, 11}},  {{-12, 12}}, {{13, 33}},
    {{21, -32}}, {{22,  -1}}, {{23, 0}},
    {{31, 13}},  {{-32, 78}}, {{33, 45}},
    {{41, 14}},  {{0,   42}}, {{43, -1}},
};
static const size_t ldc = N;

static void
printResult(void)
{
    size_t i, j, nrows;

    printf("Result:\n");

    nrows = (sizeof(C) / sizeof(cl_float2)) / ldc;
    for (i = 0; i < nrows; i++) {
        for (j = 0; j < ldc; j++) {
            printf("<%9.2f, %-9.2f> ", CREAL(C[i * ldc + j]), CIMAG(C[i*ldc + j]));
        }
        printf("\n");
    }
}

int
main(void)
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * M * sizeof(*A),
                          NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * N * sizeof(*B),
                          NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
        M * M * sizeof(*A), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0,
        M * N * sizeof(*B), B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0,
        M * N * sizeof(*C), C, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasChemm(order, side, uplo, M, N, alpha, bufA,
                         0, lda, bufB, 0, ldb, beta, bufC, 0, ldc, 1, &queue,
                         0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSsymm() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(*C),
                                  C, 0, NULL, NULL);

        /* At this point you will get the result of SYMM placed in C array. */
        printResult();
    }
    
    /* Release OpenCL events. */
    clReleaseEvent(event);
    
    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
```

结果：

```
Result:
< 41430.00, 46230.00 > <-39740.00, 87400.00 > < 48960.00, 48400.00 > 
< 41360.00, 54760.00 > <-48340.00, 90520.00 > < 32620.00, 53220.00 > 
< 28830.00, 79370.00 > <-67980.00, 77040.00 > < 13400.00, 81160.00 > 
<-24980.00, 90100.00 > <-114700.00, -43780.00> <-67560.00, 93200.00 > 
```

`clblasScopy` 是 `clBLAS` 库中的一个函数，它是 BLAS 标准中 `scopy` 函数的 OpenCL 版本。`scopy` 函数的作用是复制浮点数组。在 `clBLAS` 中，`clblasScopy` 用于将一个浮点数组复制到另一个浮点数组，这两个数组可以位于不同的内存区域。

示例代码如下：

```
/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/

#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
#include <clBLAS.h>

/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */
static const size_t N = 7;
static cl_float X[] = {
    11,
    21,
    31,
    41,
    51,
    61,
    71,
};
static const int incx = 1;

static cl_float Y[] = {
    0,
    2,
    0,
    0,
    0,
    5,
    0,
};
static const int incy = 1;


static void
printResult(void)
{
    size_t i;
    printf("\nResult:\n");

    printf(" X\n");
    for (i = 0; i < N; i++) {
			printf("\t%f\n", X[i]);
    }

    printf("Y\n");
    for (i = 0; i < N; i++) {
            printf("\t%f\n", Y[i]);
    }
}

int
main(void)
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufX, bufY;
    cl_event event = NULL;
    int ret = 0;
	int lenX = 1 + (N-1)*abs(incx);
	int lenY = 1 + (N-1)*abs(incy);

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (lenX*sizeof(cl_float)), NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (lenY*sizeof(cl_float)), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)), X, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufY, CL_TRUE, 0, (lenY*sizeof(cl_float)), Y, 0, NULL, NULL);

    /* Call clblas function. */
    err = clblasScopy( N, bufX, 0, incx, bufY, 0, incy, 1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasScopy() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufX, CL_TRUE, 0, (lenX*sizeof(cl_float)),
                                    X, 0, NULL, NULL);
        err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, (lenY*sizeof(cl_float)),
                                    Y, 0, NULL, NULL);

        /* At this point you will get the result of SSWAP placed in vector Y. */
        printResult();
    }

    /* Release OpenCL events. */
    clReleaseEvent(event);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufY);
    clReleaseMemObject(bufX);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
```

结果：

```
Result:
 X
        11.000000
        21.000000
        31.000000
        41.000000
        51.000000
        61.000000
        71.000000
Y
        11.000000
        21.000000
        31.000000
        41.000000
        51.000000
        61.000000
        71.000000
```

`clblasSgemm` 是 `clBLAS` 库中的一个函数，用于执行单精度浮点数的矩阵乘法。`Sgemm` 代表单精度（Single precision）和矩阵乘法（GEneral Matrix-Matrix multiplication）。这个函数是 BLAS 库中最基本的函数之一，广泛用于科学计算、工程模拟、数据分析和机器学习等领域。

示例代码如下：

```c++
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <clBLAS.h>
#include <sys/time.h>

#define M 320
#define N 320
#define K 320
#define ITERATIONS 300

static const clblasOrder order = clblasRowMajor;
static const cl_float alpha = 1.0f;
static const clblasTranspose transA = clblasNoTrans;
static const clblasTranspose transB = clblasNoTrans;
static const cl_float beta = 0.0f;

static cl_float A[M*K];
static cl_float B[K*N];
static cl_float C[M*N];
static cl_float result[M*N];

void initMatrix(cl_float *mat, size_t size, cl_float value) {
    for (size_t i = 0; i < size; i++) {
        mat[i] = value;
    }
}

double getCurrentTimeInMilliseconds() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}

int main(void) {
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;

    printf("[Matrix Multiply Using clBLAS] - Starting...\n");

    // Initialize matrices
    initMatrix(A, M * K, 1.0f);
    initMatrix(B, K * N, 0.01f);
    initMatrix(C, M * N, 0.0f);

    // Setup OpenCL environment
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context and command queue
    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, device, 0, &err);

    // Setup clBLAS
    clblasSetup();

    // Prepare OpenCL memory objects
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A), NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B), NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C), NULL, &err);

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, M * K * sizeof(*A), A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, K * N * sizeof(*B), B, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(*C), C, 0, NULL, NULL);

    // Perform gemm and time it
    double startTime = getCurrentTimeInMilliseconds();
    for (int i = 0; i < ITERATIONS; i++) {
        err = clblasSgemm(order, transA, transB, M, N, K,
                          alpha, bufA, 0, K,
                          bufB, 0, N, beta,
                          bufC, 0, N,
                          1, &queue, 0, NULL, &event);
        clWaitForEvents(1, &event);
    }
    double endTime = getCurrentTimeInMilliseconds();

    // Calculate performance metrics
    double elapsedTimeMs = endTime - startTime;
    double timePerIterationMs = elapsedTimeMs / ITERATIONS;
    double flops = 2.0 * M * N * K;  // 2 * M * N * K floating-point operations per matrix multiplication
    double gflops = (flops / (timePerIterationMs / 1000.0)) / 1e9;

    // Fetch results of calculations from GPU memory
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, M * N * sizeof(*result), result, 0, NULL, NULL);

    // Print performance results
    printf("MatrixA(%dx%d), MatrixB(%dx%d)\n", M, K, K, N);
    printf("clBLAS Performance = %.2f GFlop/s, Time = %.3f msec\n", gflops, timePerIterationMs);

    // Cleanup
    clReleaseEvent(event);
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    clblasTeardown();
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
```

结果：

```
[Matrix Multiply Using clBLAS] - Starting...
MatrixA(320x320), MatrixB(320x320)
clBLAS Performance = 972.25 GFlop/s, Time = 0.067 msec
```

#### 3.2.6 框架模型层

DLPrimitives-OpenCL 是一个用于在 OpenCL 上运行 PyTorch 的扩展，使得用户能够利用非 CUDA 的 GPU 进行模型训练和推理。通过该扩展，开发者可以将 PyTorch 模型部署在支持 OpenCL 的设备上，从而打破 CUDA 的限制，实现更广泛的硬件兼容性。以下是训练示例代码，展示了如何在 OpenCL 设备上执行模型的基准测试和推理。

参考仓库地址：[pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim) 

示例代码如下：

```python
###############################################################################
###
### Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
###
### MIT License, see LICENSE.TXT
###
###############################################################################
import torch

import torchvision
import json
import os
import PIL
import argparse
import time
import numpy as np
import sys
import csv

def _prof_summary(report):
    sums=dict()
    counts=dict()
    summary=[]
    for line in [v for v in report.split('\n') if v]:
       row = [v for v in line.split(' ') if v]
       name=row[0]
       val=float(row[1])
       new_val = sums.get(name,0) + val
       new_cnt =counts.get(name,0) + 1
       sums[name ] = new_val
       counts[name] = new_cnt

    for name in sums:
        summary.append((name,sums[name],counts[name]))

    summary.sort(key = lambda x:x[1])
    print("Summary:")
    print("------")
    for r in summary:
        print("%10.5f %5d %s" % ( r[1],r[2],r[0]))
    print("------")

def benchmark_model(model,batch,device,warm,iters,train,use_solver,profile):
    def _sync():
        if device.find('opencl')==0 or device.find('privateuseone')==0 or device.find('ocl')==0:
            torch.ocl.synchronize()
        elif device.find('xpu')==0:
            torch.xpu.synchronize()
        elif device.find('cuda')==0:
            torch.cuda.synchronize()

    if train:
        model.train()
    else:
        use_solver = False
        model.eval()
    #inp_cpu = torch.randn(batch,3,224,224)
    shape = (batch,3,224,224)
    inp_cpu = torch.empty(shape,dtype=torch.float32)
    torch.randn(shape,out=inp_cpu)
    total_time = 0
    total_io = 0
    total_fw = 0
    total_bw = 0
    total_zero = 0
    total_update = 0
    total_batches = 0
    total_items = 0
    print("Warming up")
    if train:
        sm = torch.nn.LogSoftmax(dim=1)
        nll = torch.nn.NLLLoss()
        lbl_cpu = torch.randint(1000,size=(batch,))
    if use_solver:
        optimizer = torch.optim.Adam(model.parameters())
    for it in range(-warm,iters):
        def run_step():
            start = time.time()
            if use_solver:
                optimizer.zero_grad()
                _sync()
                zero_point = time.time()
            else:
                zero_point = start

            inp = inp_cpu.to(device)
            if train:
                lbl = lbl_cpu.to(device)

            _sync()
            io_point = time.time()
            res = model(inp)
            if train:
                res = sm(res)
                l=nll(res,lbl)
                _sync()
                fwd_end = time.time()
                l.backward()
                _sync()
                bwd_end = time.time();
                if use_solver:
                    optimizer.step()
                    _sync()
                    solver_end = time.time()
                else:
                    solver_end = bwd_end
            else:
                res.to('cpu') 
                _sync()
                fwd_end = time.time()
                solver_end = fwd_end
                bwd_end = fwd_end
            end = time.time()
            return start,end,zero_point,io_point,fwd_end,bwd_end,solver_end
        if it == 0 and profile:
            with torch.ocl.profile(device,"prof.csv"):
                start,end,zero_point,io_point,fwd_end,bwd_end,solver_end=run_step()
        else:
            start,end,zero_point,io_point,fwd_end,bwd_end,solver_end = run_step()
        msg = ''
        if it == -warm:
            msg = 'warming up'
        elif it == 0:
            msg = 'started'
        print("Step %2d %5.3fms  %s" % (it, (end-start) * 1e3,msg))
        if it>=0:
            total_time += end-start
            total_items += batch
            total_batches += 1
            if train:
                total_fw += fwd_end - start
                total_bw += end - fwd_end
                total_io += io_point - zero_point
                total_zero += zero_point - start
                total_update += solver_end - bwd_end
    print("Time per item  %1.3f ms" %(total_time / total_items *1e3))
    if train:
        print("Time fwd batch  %1.3f ms" %(total_fw / total_batches *1e3))
        print("Time bwd batch  %1.3f ms" %(total_bw / total_batches *1e3))
        print("Time io  batch  %1.3f ms" %(total_io / total_batches *1e3))
        print("Time zro batch  %1.3f ms" %(total_zero / total_batches *1e3))
        print("Time opt batch  %1.3f ms" %(total_update  / total_batches *1e3))

    print("Time per batch %1.3f ms" %(total_time / total_batches *1e3))

def export_model(model,batch,path,opset,ir,train):
    inp = torch.randn(batch,3,224,224)
    model.eval()
    if train:
        extra =dict( training=torch.onnx.TrainingMode.TRAINING,do_constant_folding=False)
    else:
        extra = dict(do_constant_folding=True)
    torch.onnx.export(model,inp,path,input_names = ["data"],output_names=["prob"],opset_version=opset,**extra)
    import onnx
    #from onnx import version_converter
    model = onnx.load_model(path)
    model.ir_version = ir
    onnx.save(model, path)
    
def predict_on_images(model,images,device,config):
    tw = 224
    th = 224
    mean = config['mean']
    std = config['std']
    classes = config['class_names']
    csv = []
    model.eval()
    image = torch.zeros((len(images),3,th,tw),dtype=torch.float32)
    for i,path in enumerate(images):
        img = PIL.Image.open(path)
        npimg = np.array(img).astype(np.float32) * (1.0 / 255)
        h = npimg.shape[0]
        w = npimg.shape[1]
        assert h>=th
        assert w>=tw
        assert npimg.shape[2] == 3
        fact = 1.0 / np.array(std)
        off  = -np.array(mean) * fact
        dr = (h - th) // 2
        dc = (w - tw) // 2
        for k in range(3):
            image[i,k,:,:] = torch.from_numpy(npimg[dr:dr+th,dc:dc+tw,k] * fact[k] + off[k])
    image = image.to(device)
    res = model(image)
    for i in range(len(images)):
        index = torch.argmax(res[i]).item()
        csv.append([path,str(index),classes[index]] + ['%8.6f' % v for v in res[i].tolist()])
    with open('report.csv','w') as f:
        for row in csv:
            line = ','.join(row) + '\n'
            f.write(line)
            sys.stdout.write(','.join(row[0:10] + ['...']) + '\n')
        
def get_config():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(base_path + '/examples/cpp/imagenet_predict_config.json','r') as f:
        cfg = json.load(f)
    return cfg

def main(args):
    m = getattr(torchvision.models,args.model)(weights = 'DEFAULT')
    #print("Mean",m.bn1.running_mean.tolist()[:4])
    #print("Var",m.bn1.running_var.tolist()[:4])
    #print("W",m.bn1.weight.tolist()[:4])
    #print("B",m.bn1.bias.tolist()[:4])
    if args.export:
        export_model(m,args.batch,args.export,args.onnx_opset,args.onnx_ir,args.train)
    m.to(args.device)
    if args.images:
        with torch.no_grad():
            predict_on_images(m,args.images,args.device,get_config())
    if args.benchmark:
        if args.train:
            benchmark_model(m,args.batch,args.device,args.warm,args.iters,args.train,args.solver,args.profile)
        else:
            with torch.no_grad():
                benchmark_model(m,args.batch,args.device,args.warm,args.iters,args.train,False,args.profile)

if __name__ == '__main__': 
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='vgg16')
    p.add_argument('--device',default='cuda')
    p.add_argument('--export')
    p.add_argument('--solver',action='store_true')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--train',action='store_true')
    p.add_argument('--profile',action='store_true',default=False)
    p.add_argument('--onnx-opset',default=9,type=int)
    p.add_argument('--onnx-ir',default=3,type=int)
    p.add_argument('--batch',default=16,type=int)
    p.add_argument('--warm',default=5,type=int)
    p.add_argument('--iters',default=20,type=int)
    p.add_argument('images',nargs='*')
    r = p.parse_args()
    if r.device.find('ocl')==0 or r.device.find('privateuseone')==0:
        import pytorch_ocl
        if r.profile:
            torch.ocl.enable_profiling(r.device)
    if r.device.find('xpu')==0:
        import intel_extension_for_pytorch
    main(r)
```

结果：

```
         //    net batch time
             alexnet 64 24.543
            resnet18 64 70.040
            resnet50 32 113.758
      convnext_small 16 155.833
               vgg16 16 104.042
         densenet161 16 142.568
        mobilenet_v2 32 56.262
  mobilenet_v3_small 64 35.727
  mobilenet_v3_large 64 87.085
     resnext50_32x4d 32 144.684
     wide_resnet50_2 32 190.366
          mnasnet1_0 32 51.156
     efficientnet_b0 32 85.117
      regnet_y_400mf 64 77.130
```

#### 3.2.7 对比与思考

### 3.3 SYCL (DPC++) (NVIDIA)

#### 3.3.1 技术栈架构

**1. 系统软件层**
- 后端驱动程序：
  - OpenCL 驱动：为支持 OpenCL 的设备提供底层支持
  - CUDA 驱动：允许在 NVIDIA GPU 上运行 SYCL 代码
  - Level Zero 驱动：Intel 的低级硬件抽象层，为 Intel GPU 提供直接访问
- 硬件抽象层：
  - 提供统一的接口，隐藏不同后端的复杂性
  - 允许 SYCL 在多种硬件平台上运行，包括 CPU、GPU 和 FPGA

**2. 运行时环境层**
- SYCL Runtime：
  - 管理设备发现、内存分配和数据传输
  - 处理任务调度和执行
  - 实现异步执行模型和事件同步
  - 提供错误处理和异常管理
  - 支持设备选择和上下文管理

**3. 编程模型和语言层**
- SYCL C++：
  - 基于现代 C++ 标准（C++17 或更高）
  - 提供单源编程模型，主机和设备代码在同一文件中
  - 使用模板和 lambda 表达式简化并行编程
  - 支持数据并行和任务并行编程模型
- DPC++ (Data Parallel C++)：
  - Intel 的 SYCL 实现和扩展
  - 增加了额外的功能，如统一共享内存（USM）和子组功能
  - 提供与 Intel 硬件的深度集成和优化

**4. 计算库层**
- SYCL-BLAS：
  - 提供 BLAS（基础线性代数子程序）的 SYCL 实现
  - 支持向量和矩阵操作的高性能计算
  - 针对不同硬件后端优化
- oneDPL (oneAPI DPC++ Library)：
  - 提供并行算法和容器
  - 实现了许多标准模板库（STL）的并行版本
- oneDNN (oneAPI Deep Neural Network Library)：
  - 深度学习原语的高性能实现
  - 支持卷积、池化等常见神经网络操作

**5. 框架模型层**

- TensorFlow with SYCL：
  - 通过 SYCL 后端支持，允许 TensorFlow 模型在多种硬件上运行
- PyTorch with SYCL：
  - 集成 SYCL 支持，提供 PyTorch 在异构系统上的加速



##### 关系解析
![alt text](1261726037696_.pic.jpg)

SYCL作为一个统一的高级抽象层，连接了多种底层计算技术，包括PyTorch、OpenCL和CUDA。在与PyTorch的集成方面，SYCL提供了计算加速和程序优化的能力，允许开发者利用SYCL的并行计算能力来增强PyTorch模型。对于OpenCL，SYCL简化了其使用复杂性，提供了更高级的抽象，同时保持了对OpenCL底层功能的访问能力。在CUDA方面，SYCL允许代码在NVIDIA GPU上运行，同时保持跨平台兼容性，为开发者提供了更大的灵活性。SYCL-BLAS作为一个重要组件，提供了高效的线性代数运算，支持各种硬件平台的优化。SYCL C++/DPC++扩展了C++标准，提供了更灵活的编程模型，特别适合Intel架构。SYCL Runtime作为核心组件，管理设备执行、内存同步、任务调度等关键功能，确保了跨平台的一致性和高效性。这种架构设计使SYCL能够在保持高性能的同时，提供了跨多种硬件平台的统一编程模型，大大简化了异构计算的开发复杂度，使开发者能够更容易地利用不同的加速器技术，同时保持代码的可移植性和效率。

#### 3.3.2 系统软件层

#### 3.3.3 运行时环境层

#### 3.3.4 编程模型和语言层

#### 3.3.5 计算库层

#### 3.3.6 框架模型层

#### 3.3.7 对比与思考

### 3.4 Triton (NVIDIA)

#### 3.4.1 技术栈架构
Triton 是一个创新的开源项目，旨在简化 GPU 编程并提高计算性能。它提供了一种高级抽象，使开发者能够更容易地编写高效的 GPU 内核，而无需深入了解底层硬件细节。Triton 的设计理念是在保持高性能的同时，提供更好的可读性和可维护性。


##### 关系解析
![alt text](1251726020750_.pic-1.jpg)


#### 3.4.2 Triton相关技术解析


Triton 引入了几个关键概念：

- Triton DSL（领域特定语言）：Triton 提供了一种特定于 GPU 编程的语言，它是 Python 的一个子集，增加了一些特定于并行计算的原语。

- 自动调优：Triton 能够自动选择最佳的执行参数，如线程块大小和内存访问模式。

- 多维张量操作: Triton 原生支持多维张量操作，使得复杂的数学运算变得简单。

- 动态形状支持:与传统 CUDA 编程不同，Triton 支持动态形状的输入，增加了代码的灵活性。


![alt text](1331726123229_.pic.jpg)

##### Triton 的工作原理

Triton 的工作原理可以分为以下几个关键步骤：
 - 代码分析：Triton 编译器分析用 Triton DSL 编写的代码。
 - 中间表示生成：将代码转换为 MLIR（Multi-Level Intermediate Representation）。
 - 优化：在 MLIR 层面进行各种优化。
 - 代码生成：将优化后的 MLIR 转换为目标硬件的机器代码（如 PTX 或 AMDGPU ISA）。
 - 运行时执行：使用相应的 GPU API 加载和执行生成的代码。

##### Triton 的编译流程

Triton 的编译流程是其核心优势之一，包括以下主要阶段：

1. **Triton DSL → MLIR**
   - 解析 Triton DSL 代码
   - 生成初始的 MLIR 表示

2. **MLIR 优化**
   - 执行特定于 Triton 的优化pass
   - 应用通用的 MLIR 优化

3. **MLIR → LLVM IR**
   - 将优化后的 MLIR 转换为 LLVM IR

4. **LLVM IR → 目标代码**
   - 使用 LLVM 后端生成目标特定的机器代码（如 PTX）

5. **JIT 编译**
   - 在运行时即时编译生成的代码

这个流程允许 Triton 在保持高级抽象的同时，生成高度优化的机器代码。

##### 与 GPU Runtime 和 API 的对接

Triton 通过多层抽象与不同的 GPU 平台进行交互：

1. ***CUDA Driver API***
- 使用低级 API 如 `cuModuleLoad` 和 `cuLaunchKernel` 加载和执行 PTX 代码。

2. ***CUDA Runtime API***
- 利用更高级的 API 如 `cudaLaunchKernel` 简化内核启动过程。

3. ***ROCm 和 HIP API***
- 为 AMD GPU 提供支持，使用 HIP API 进行交互。

4. ***具体实现细节***
- 代码生成：生成适合目标平台的代码（PTX 或 AMDGPU ISA）。
- 运行时集成：创建封装底层 API 调用的 GPU Driver 对象。
- 内核加载与启动：使用相应的 API 加载编译好的 GPU 代码并启动内核。
- 结果获取与错误处理：同步执行结果，处理可能的错误。

##### Triton 的抽象层

Triton 提供了多个抽象层，以简化跨平台 GPU 编程：

设备抽象
- 定义通用的 Device 接口，隐藏不同 GPU 架构的细节。

内存管理抽象
- 提供统一的内存分配和数据传输接口。

内核启动抽象
- 简化不同平台上的内核配置和启动过程。

编程模型抽象
- 提供统一的编程模型，使开发者能够编写可移植的代码。

##### Triton 与 PyTorch 的集成

Triton 可以与 PyTorch 无缝集成，为深度学习模型提供性能优化：

自定义 CUDA 内核
- 使用 Triton 编写高效的自定义操作，集成到 PyTorch 模型中。

性能关键操作的优化
- 针对特定的计算密集型操作，如矩阵乘法，使用 Triton 实现高性能版本。

##### Triton 的优势与局限性

###### 优势
1. 简化 GPU 编程，提高开发效率。
2. 自动优化，减少手动调优的需求。
3. 良好的可移植性，支持多种 GPU 架构。
4. 与 PyTorch 等深度学习框架的无缝集成。

###### 局限性
1. 学习曲线可能较陡，特别是对于不熟悉 GPU 编程的开发者。
2. 在某些极端情况下，手动优化的 CUDA 代码可能仍然更快。
3. 生态系统相对较新，社区支持和工具链还在发展中。


#### 3.4.2 技术栈架构

1. **系统软件层**
   - **NVIDIA GPU 驱动**：为 GPU 提供基本的系统级支持。
   - **CUDA Driver API**：低级 API，提供对 GPU 的直接控制。
     - 允许直接管理设备、内存分配和程序执行。
     - 适用于需要细粒度控制的高级应用。
     - 提供与 NVIDIA GPU 硬件交互的底层接口。

2. **运行时环境层**
   - **CUDA Runtime API**：高级 API，简化了 GPU 编程，自动管理许多底层细节。
     - 提供更高级的抽象，简化了 GPU 的使用。
     - 自动处理上下文管理和程序加载等任务。
     - 更适合一般开发者使用，提供了更好的易用性。

3. **编程模型和语言层**
   - **Triton DSL (领域特定语言)**：扩展了 Python，允许开发者编写在 GPU 上运行的并行程序。
     - 允许在 CPU 和 GPU 上混合编程。
     - 使用 Triton 特定语法定义 GPU 函数。
     - 通过方言（Dialect）提供优化的操作和功能。

4. **计算库层**
   - **Triton 实现的算子库**：提供高性能的计算内核，专门针对各种深度学习操作进行优化。
     - 针对特定操作的高效实现，如矩阵运算。

5. **框架模型层**
   - **PyTorch**：支持动态计算图的深度学习框架，通过 `torch.cuda` 模块提供 CUDA 功能。
     - 自动管理 GPU 内存，支持 GPU 和 CPU 之间的数据转移。
   - **TensorFlow**：支持静态和动态计算图的深度学习框架。
     - 通过 XLA 编译器优化 GPU 代码执行，提供高级 API 来简化 CUDA API 的使用。

![alt text](1251726020750_.pic.jpg)

#### 3.4.3 系统软件层
Triton 通过使用 CUDA Driver API 与底层 GPU 进行交互。具体流程如下：

- Triton 生成的代码将被编译为 PTX（Parallel Thread Execution）代码，用于 NVIDIA GPU。
- 通过 CUDA Driver API（例如 `cuModuleLoad`, `cuLaunchKernel` 等）来加载和执行这些 PTX 代码。
#### 3.4.4 运行时环境层
Triton 的设计使得它能够灵活地与 GPU 进行交互，涉及多个层次的抽象和转换。

除了 CUDA Driver API，Triton 还可以利用 CUDA Runtime API，这是建立在 Driver API 之上的更高级别接口，常见的操作包括：

- 使用 `cudaLaunchKernel` 来启动内核。
- 为 AMD GPU 提供支持，使用 ROCm 与 HIP API 进行交互。
#### 3.4.5 编程模型和语言层

Triton 语言是为高性能计算而设计的领域特定语言 (DSL)，其编译流程如下：

1. **代码生成**：Triton 编译器分析 `@triton.jit` 装饰的函数。
2. **多级中间表示（MLIR）**：转换生成适合 Triton 的 MLIR 表示。
3. **LLVM IR**：再进一步优化和转换为 LLVM IR。
4. **目标代码**：编译为 PTX 或其他目标代码。

Triton 提供了高性能的计算库，开发者可以利用这些库进行高效操作。例如，标准的Add（向量加法）、 GEMM（矩阵乘法）等操作可以使用 Triton 的编程模型实现，利用自定义内存访问模式和自动调优功能获得更佳性能。

参考仓库地址：[triton](https://github.com/triton-lang/triton) 

向量加法的实现示例代码如下：

```python
"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:

def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True, save_path="output")

```

结果：

```
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
The maximum difference between torch and triton is 0.0
vector-add-performance:
           size      Triton       Torch
0        4096.0   12.000000   12.000000
1        8192.0   24.000000   24.000000
2       16384.0   44.846717   44.521738
3       32768.0   76.800002   76.800002
4       65536.0  148.048195  151.703707
5      131072.0  219.428568  222.407245
6      262144.0  341.333321  365.442364
7      524288.0  433.534740  409.600010
8     1048576.0  506.069512  491.520012
9     2097152.0  534.260858  546.133325
10    4194304.0  564.965515  564.965515
11    8388608.0  606.522314  603.092009
12   16777216.0  620.214515  611.949793
13   33554432.0  632.084809  627.310727
14   67108864.0  635.243943  642.034476
15  134217728.0  638.078725  633.441130
```


![alt text](1991727424657_.pic.jpg)
下面将实现一个融合的 softmax 操作，该操作在处理特定类型的矩阵时，性能显著优于 PyTorch 的原生实现。具体而言，当矩阵的行可以适应 GPU 的 SRAM 时，融合内核可以减少内存访问并提高计算效率。通过这个例子，我们将学习内核融合的好处以及 Triton 中的归约操作。

融合的 softmax实现示例代码如下：

```python
"""
Fused Softmax
=============

In this tutorial, you will write a fused softmax operation that is significantly faster
than PyTorch's native op for a particular class of matrices: those whose rows can fit in
the GPU's SRAM.

In doing so, you will learn about:

* The benefits of kernel fusion for bandwidth-bound operations.

* Reduction operators in Triton.

"""

# %%
# Motivations
# -----------
#
# Custom GPU kernels for elementwise additions are educationally valuable but won't get you very far in practice.
# Let us consider instead the case of a simple (numerically stabilized) softmax operation:

import torch

import triton
import triton.language as tl
from triton.runtime import driver

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch

    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    """
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret

# %%
# When implemented naively in PyTorch, computing :code:`y = naive_softmax(x)` for :math:`x \in R^{M \times N}`
# requires reading :math:`5MN + 2M` elements from DRAM and writing back :math:`3MN + 2M` elements.
# This is obviously wasteful; we'd prefer to have a custom "fused" kernel that only reads
# X once and does all the necessary computations on-chip.
# Doing so would require reading and writing back only :math:`MN` bytes, so we could
# expect a theoretical speed-up of ~4x (i.e., :math:`(8MN + 4M) / 2MN`).
# The `torch.jit.script` flags aims to perform this kind of "kernel fusion" automatically
# but, as we will see later, it is still far from ideal.

# %%
# Compute Kernel
# --------------
#
# Our softmax kernel works as follows: each program loads a set of rows of the input matrix X strided by number of programs,
# normalizes it and writes back the result to the output Y.
#
# Note that one important limitation of Triton is that each block must have a
# power-of-two number of elements, so we need to internally "pad" each row and guard the
# memory operations properly if we want to handle any possible input shapes:


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

# %%
# We can create a helper function that enqueues the kernel and its (meta-)arguments for any given input tensor.

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y

# %%
# Unit Test
# ---------

# %%
# We make sure that we test our kernel on a matrix with an irregular number of rows and columns.
# This will allow us to verify that our padding mechanism works.

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

# %%
# As expected, the results are identical.

# %%
# Benchmark
# ---------
#
# Here we will benchmark our operation as a function of the number of columns in the input matrix -- assuming 4096 rows.
# We will then compare its performance against (1) :code:`torch.softmax` and (2) the :code:`naive_softmax` defined above.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(show_plots=True, print_data=True, save_path="output")
```

结果：

```
softmax-performance:
          N      Triton       Torch
0     256.0  398.912462  524.814932
1     384.0  417.129147  562.317814
2     512.0  485.794808  558.852447
3     640.0  471.350087  548.172858
4     768.0  494.683107  538.367338
5     896.0  494.294775  536.426279
6    1024.0  513.330526  562.354982
7    1152.0  509.280093  571.505647
...
96  12544.0  597.487978  632.835908
97  12672.0  595.450177  628.906688
```
![alt text](1971727424637_.pic.jpg)

矩阵乘法的实现示例代码如下：

```python
import torch
import triton
import triton.language as tl
import time

# Define matrix multiplication kernel using Triton
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // (N // BLOCK_N)
    col = pid % (N // BLOCK_N)
    
    offs_m = row * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = col * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accum += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    
    c_ptrs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accum)

def matmul(a, b):
    num_iters = 300
    M, K = a.shape
    N = b.shape[1]
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    # Compile and run Triton kernel
    grid = (M // 32, N // 32)
    
    start = time.time()
    print("[Matrix Multiply Using Triton] - Starting...")
    print(f"MatrixA({M},{K}), MatrixB({K},{N})")
    
    for _ in range(num_iters):
        matmul_kernel[grid](a, b, C, M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)

    torch.cuda.synchronize()
    end = time.time()

    # Calculate performance metrics
    elapsed_time = end - start
    time_per_iteration = elapsed_time * 1000 / num_iters
    flops = 2.0 * M * N * K * num_iters
    gflops = (flops / elapsed_time) / 1e9

    # Output performance results
    print(f"Triton Performance= {gflops:.2f} GFlop/s, Time= {time_per_iteration:.3f} msec")
    return C

# Matrix sizes
M, N, K = 320, 320, 320

# Initialize matrices
A = torch.randn((M, K), dtype=torch.float16, device='cuda')
B = torch.randn((K, N), dtype=torch.float16, device='cuda')

# Call the matmul function
C = matmul(A, B)
print(f"Output matrix C: {C}")
```

结果：

```
[Matrix Multiply Using Triton] - Starting...
MatrixA(320,320), MatrixB(320,320)
Triton Performance= 79.84 GFlop/s, Time= 0.821 msec
Output matrix C: tensor([[  0.1220,  -4.0168, -11.4398,  ...,   1.5115,  -4.4500,  10.5483],
        [ 18.3915,  21.7275, -15.4414,  ...,  -8.9633,  32.6608,  27.5713],
        [-31.2961,   7.7287,   8.6794,  ...,  10.2873,  -3.2942,  26.0596],
        ...,
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,   0.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,  ...,   0.0000,   0.0000,   0.0000]],
       device='cuda:0')
```

#### 3.4.7 框架模型层

Triton 可以与 PyTorch 框架无缝集成，虽然 PyTorch 模型不会直接转换为 Triton，但可以利用 Triton 编写自定义的 CUDA 核心，从而优化特定的操作。这种方式让开发者可以在 PyTorch 中使用 Triton 优化的操作，提升性能。

例如，在 PyTorch 模型中包装 Triton 核心的代码：

```python
class MyModel(torch.nn.Module):
    def forward(self, x, y):
        z = triton_add_wrapper(x, y)
        return z
```

Unsloth 是一个高效的库，使用 Triton 编写的算子，能够实现高性能的模型训练和推理，且没有准确性损失。下面是使用 Unsloth 的 FastLanguageModel 来加载一个预训练的 LLaMA 3 模型并进行推理的示例代码：

```python
import time 
import torch
from unsloth import FastLanguageModel

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/aii-works/llama3/Meta-Llama-3-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

FastLanguageModel.for_inference(model) 
       
inputs = tokenizer(
[
    alpaca_prompt.format(
        # "Continue the fibonnaci sequence.", # instruction
        "Q:",
        "Name the planets in the solar system?",
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

iterations = 10
with torch.no_grad():
    for _ in range(5):
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True) 
t_start = time.time()
for _ in range(iterations):
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True) 
elapsed_time = time.time() - t_start
latency = elapsed_time / iterations * 1000
FPS = 1000 / latency

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(f"FPS: {FPS:.2f}")
```

结果：

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
==((====))==  Unsloth 2024.8: Fast Llama patching. Transformers = 4.44.2.
   \\   /|    GPU: NVIDIA GeForce RTX 4080 SUPER. Max memory: 15.695 GB. Platform = Linux.
O^O/ \_/ \    Pytorch: 2.4.0. CUDA = 8.9. CUDA Toolkit = 12.1.
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.27.post2. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:02<00:00,  1.76it/s]
/home/aii-works/llama3/Meta-Llama-3-8B-Instruct does not have a padding token! Will use pad_token = <|reserved_special_token_250|>.
Unsloth 2024.8 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Q:

### Input:
Name the planets in the solar system?

### Response:
The eight planets in our solar system, in order from the Sun, are:

1. Mercury
2. Venus
3. Earth
4. Mars
5. Jupiter
6. Saturn
7. Uranus
8. Neptune

Note: Pluto was previously considered a planet, but in 2006,
FPS: 0.89
```

#### 3.4.8 对比与思考

### 3.5 Apache TVM (NVIDIA)

#### 3.5.1 技术栈架构
TVM 支持的硬件概述
下图显示了 TVM 当前支持的硬件后端：

![img](https://github.com/dmlc/web-data/raw/main/tvm/tutorial/tvm_support_list.png)

**（此图为官网图，需重绘）**

#### 3.5.2 系统软件层

#### 3.5.3 运行时环境层

编写了一个程序，列出系统中可用的 CUDA 设备，获取设备的名称、计算能力和全局内存大小等信息。

- 检查 CUDA 设备: 定义 `check_cuda` 函数以尝试获取第一个 CUDA 设备并收集其信息。
- 获取设备信息: 包括计算能力、最大线程数、共享内存等。
- 异常处理: 捕获并打印获取设备信息时的错误。
- 输出设备信息: 打印设备的详细信息和名称。

代码：

```python
import tvm
from tvm.contrib import nvcc

# 检查 TVM 是否支持 CUDA 并返回设备详细信息
def check_cuda():
    try:
        # 尝试获取 CUDA 设备
        device = tvm.cuda(0)

        # 获取设备的详细信息
        device_info = {
            "compute_capability": device.compute_version,  # 计算能力
            "max_threads_per_block": device.max_threads_per_block,  # 每个块的最大线程数
            "max_shared_memory_per_block": device.max_shared_memory_per_block,  # 每个块的最大共享内存
            "multi_processor_count": device.multi_processor_count,  # 多处理器数量
            "warp_size": device.warp_size,  # warp 大小
            "total_global_memory": device.total_global_memory,  # 总全局内存
        }
        print("CUDA check success")
        print("Device Info:")
        for key, value in device_info.items():
            print(f"  {key}: {value}")
        return device_info
    except Exception as e:
        print(f"CUDA check failed: {e}")
        return None

check_cuda()

# 获取当前可用的设备
dev = tvm.cuda(0)  # 获取第一个 GPU 设备

# 输出设备名称
device_name = dev.device_name
print("Device Name:", device_name)

# 获取设备的详细信息
device_info = {
    "Device Type": dev.device_type,
    "Device ID": dev.device_id,  # 使用 device_id 替代 device_index
}
# 输出设备详细信息
for key, value in device_info.items():
    print(f"{key}: {value}")
```

结果：

```
CUDA check success
Device Info:
  compute_capability: 8.9
  max_threads_per_block: 1024
  max_shared_memory_per_block: 49152
  multi_processor_count: 80
  warp_size: 32
  total_global_memory: 16852844544
Device Name: NVIDIA GeForce RTX 4080 SUPER
Device Type: 2
Device ID: 0
```

#### 3.5.4 编程模型和语言层

[此处填写 CUDA 编程模型和语言层内容]

#### 3.5.5 计算库层

TVM 仓库的根目录包含以下几个子目录：

- **src**：存放用于算子编译和部署运行时的 C++ 代码。
- **src/relay**：实现了 Relay，这是一个为深度学习框架提供的新功能 IR。
- **python**：提供 Python 前端，用于封装 src 中实现的 C++ 函数和对象。
- **src/topi**：定义标准神经网络算子的计算和后端调度。

**src/relay** 是负责管理计算图的组件，其中图结构中的节点通过 **src** 其余部分提供的基础设施进行编译和执行。**python** 为 C++ API 和执行编译的驱动代码提供了 Python 绑定。与节点对应的算子在 **src/relay/op** 中注册，而算子的实现则在 **topi** 中，使用的编程语言包括 C++ 和 Python。

其中：

- **IR（Intermediate Representation）**：一种中间表示形式，用于编译过程中的高级代码表示。
- **算子（Operator）**：在深度学习中，算子通常指代执行特定计算的函数，比如卷积、矩阵乘等。
- **调度（Schedule）**：定义了算子如何在硬件上执行的策略，包括循环的嵌套结构、并行化、向量化等。

向量加法示例：

```
n = 1024
A = tvm.te.placeholder((n,), name='A')
B = tvm.te.placeholder((n,), name='B')
C = tvm.te.compute(A.shape, lambda i: A[i] + B[i], name="C")
```

在 `python/tvm/te/tensor.py` 中定义的 `A`、`B` 和 `C` 的类型都是 `tvm.tensor.Tensor`。这些 Python Tensor 是由 C++ Tensor 支持的，其实现位于 `include/tvm/te/tensor.h` 和 `src/te/tensor.cc` 文件中。在 TVM 中，所有的 Python 类型都可以视为与其底层 C++ 类型具有相同名称的句柄。

从以下 Python Tensor 类型的定义中可以看出，`tvm.tensor.Tensor` 是 `Object` 的一个子类。

```
@register_object
class Tensor(Object, _expr.ExprOp):
    """Tensor object, to construct, see function.Tensor"""
  
    def __call__(self, *indices):
       ...
```

- 在 TVM 中，每个 `Tensor` 对象都有一个与之关联的 `Operation` 对象。`Tensor` 是在计算过程中存储数据的多维数组，而 `Operation` 表示对一个或多个 `Tensor` 进行操作的计算。这两个概念在代码中有明确的实现，相关定义分别在 `python/tvm/te/tensor.py`、`include/tvm/te/operation.h` 和 `src/tvm/te/operation` 目录下。

- 每个 `Tensor` 对象都可以看作是其相应的 `Operation` 的输出，这意味着通过执行某个 `Operation` 可以生成一个 `Tensor`。

- `Operation` 对象提供了一个 `input_tensors()` 方法，这个方法返回一个输入 `Tensor` 的列表。这使得开发者能够跟踪不同 `Operation` 之间的依赖关系，了解一个 `Operation` 需要哪些输入 `Tensor`，以及这些输入 `Tensor` 是由哪些其他 `Operation` 产生的。

- 在计算图中，当我们想要调度某个计算时，需要将输出张量（例如上面提到的 `C` 张量）对应的 `Operation` 对象传递给 `python/tvm/te/schedule.py` 中的 `tvm.te.create_schedule()` 函数`create_schedule()` 函数负责生成计算的调度策略，以优化计算的执行。这是构建高效计算图的重要步骤，因为它允许对计算的执行顺序和方式进行控制，从而提高性能。

```
S = tvm.te.create_schedule(C.op)
```

函数被映射到 `include/tvm/schedule.h` 中的 C++ 函数。

```
inline Schedule create_schedule(Array<Operation> ops) {
    return Schedule(ops);
}
```

- 在 TVM 中，调度由多个 `Stage` 和输出的 `Operation` 组成。每个 `Stage` 代表一个 `Operation` 的计算过程。
- 以“向量加法”（Vector Add）为例，假设有两个占位符 `Operation` 和一个计算 `Operation`，那么这个调度（`s`）将包含三个阶段（`Stage`）。

- 每个 `Stage` 存储有关循环嵌套的信息，包括：循环嵌套结构：描述了如何将计算划分为多个循环的结构。循环类型：标识每个循环的执行方式，比如：Parallel（并行）：表示该循环可以在多个线程中并行执行。Vectorized（向量化）：表示该循环将数据分块处理，以提高效率。Unrolled（展开）：表示将循环展开为多个相同的操作，以减少循环开销。位置：指明在下一个 `Stage` 的循环嵌套中执行该计算的位置（如果有嵌套的话）。create_schedule() 函数的作用：`create_schedule()` 函数用于创建默认的调度。这个调度提供了基础的计算顺序和结构。默认的调度通常会调用 `tvm.build(...)` 函数来生成可执行的代码。

- 为了使调度可以在 GPU 上运行，需要为调度中的 `Stage` 绑定必要的线程。这一步骤是非常重要的，因为 GPU 的并行计算能力依赖于对线程的有效管理和分配。
- 通过线程绑定，开发者可以控制计算的并行性，从而充分利用 GPU 的硬件资源，以实现更高的性能。

```
target = "cuda"
bx, tx = s[C].split(C.op.axis[0], factor=64)
s[C].bind(bx, tvm.te.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.te.thread_axis("threadIdx.x"))
fadd = tvm.build(s, [A, B, C], target)
```

- `target = "cuda"` 指定了目标平台是CUDA，意味着生成的代码将在GPU上运行。
- `split`和`bind`是调度操作，用于优化并行执行。`split`将计算操作的循环分割成更小的部分，`bind`将这些分割的部分绑定到GPU的线程和块上。
- `tvm.build`函数接受调度、输入和输出Tensor以及目标平台，然后返回一个可以在该平台上运行的模块。

tvm.build() 函数：
- `tvm.build()` 函数定义在 `python/tvm/driver/build_module.py` 中。它的主要作用是接收一个调度（schedule）、输入和输出的 `Tensor`，以及一个目标设备（target），然后返回一个 `tvm.runtime.Module` 对象。返回的 `tvm.runtime.Module` 对象包含一个可以通过函数调用的已编译函数，这意味着用户可以直接调用这个编译后的函数进行计算，而无需关心底层实现细节。

- `tvm.build()` 的过程可以分为两个主要步骤：降级：降级过程将高级、初始的循环嵌套结构转化为最终的底层中间表示（IR）。这一过程是由 `tvm.lower()` 函数完成的，`tvm.lower()` 也定义在 `python/tvm/build_module.py` 中。降级的第一步是进行边界推断，确定每个循环的迭代范围，以便在生成 IR 时确保计算的正确性。随后，`tvm.lower()` 会创建一个初始的循环嵌套结构，以便更好地表达计算的逻辑和顺序。代码生成：在降级完成后，接下来的步骤是根据底层的 IR 生成目标机器代码。这一过程涉及将 IR 转换为特定硬件可以理解和执行的机器代码。

- 降级的过程有助于将更高级的计算抽象（例如高层的循环结构和调度策略）转化为更为底层的表示，使得后续的代码生成过程能够更加有效地针对特定硬件进行优化。通过将计算表示降级到 IR，TVM 能够更灵活地进行优化并适配多种硬件目标

```
def lower(sch,
          args,
          name="default_function",
          binds=None,
          simple_mode=False):
   ...
   bounds = schedule.InferBound(sch)
   stmt = schedule.ScheduleOps(sch, bounds)
   ...
```

边界推断（Bound Inference）：
- 边界推断是一个关键的过程，它用于推断所有循环的边界和中间缓冲区的大小。这对于生成有效的代码和优化计算非常重要。
- 如果目标是 CUDA 后端，并且使用了共享内存，边界推断将自动确定所需的最小共享内存尺寸。这一过程确保了在运行时可以有效利用共享内存，从而提高计算性能。

边界推断的实现：边界推断的实现代码位于以下文件中：
- `src/te/schedule/bound.cc`
- `src/te/schedule/graph.cc`
- `src/te/schedule/message_passing.cc`

- 这些实现文件负责具体的边界推断算法和逻辑，包括如何根据调度信息推断出循环的边界和缓冲区的大小。

ScheduleOps() 的作用：
- `stmt` 是 `ScheduleOps()` 函数的输出，表示一个初始的循环嵌套结构。这个结构是调度的基础，反映了计算中循环的组织方式。
- 如果调度过程中已经应用了 `reorder` 或 `split` 等原语，则 `stmt` 将反映这些变化，确保生成的初始循环结构与应用的调度操作一致。
- `ScheduleOps()` 函数的定义位于 `src/te/schedule/schedule_ops.cc` 中。

接下来，对 `stmt` 在 `src/tir/pass` 子目录下进行降级处理。

```
...
stmt = ir_pass.VectorizeLoop(stmt)
...
stmt = ir_pass.UnrollLoop(
    stmt,
    cfg.auto_unroll_max_step,
    cfg.auto_unroll_max_depth,
    cfg.auto_unroll_max_extent,
    cfg.unroll_explicit)
...
```

- 在降级完成后，`build()` 函数负责从降级后的函数生成特定目标的机器代码。这一步是将中间表示（IR）转化为实际可执行的代码。
- 如果目标是 x86 架构，生成的代码将包含 SSE（Streaming SIMD Extensions）或 AVX（Advanced Vector Extensions）指令，以优化计算性能。
- 如果目标是 CUDA，生成的代码将包含 PTX（Parallel Thread Execution）指令，这是 NVIDIA 的一种中间表示，用于描述并行计算的指令。

- 除了生成目标专用的机器代码，TVM 还会生成一段宿主机代码。这部分代码负责执行一些重要的任务，如内存管理和内核启动等。宿主机代码确保了生成的内核能够在目标设备上正确运行并管理资源。

- 代码生成的具体实现是在 `build_module()` 函数中完成的，该函数定义在 `python/tvm/target/codegen.py` 中。这个 Python 函数负责协调代码生成的各个环节。
- 在 C++ 端，代码生成的实现细节位于 `src/target/codegen` 子目录中。这里包含了许多与代码生成相关的实现和优化。

- `build_module()` 函数最终会调用 C++ 端的 `Build()` 函数，后者位于 `src/target/codegen/codegen.cc` 中。`Build()` 函数负责将具体的代码生成逻辑实现，完成从中间表示到目标机器代码的转换。

```
TVM_REGISTER_GLOBAL("codegen.build_cuda")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = BuildCUDA(args[0]);
});
```

- `BuildCUDA()` 函数使用定义在 `src/codegen/codegen_cuda.cc` 中的 `CodeGenCUDA` 类，从降级的 IR（中间表示）中生成 CUDA 内核源代码。这意味着它将高层的计算表示转化为适合在 NVIDIA GPU 上执行的 CUDA 代码。

- 生成的 CUDA 内核源代码随后通过 NVRTC（NVIDIA Runtime Compilation）进行编译。NVRTC 是 NVIDIA 提供的一个库，允许在运行时编译 CUDA 程序，方便动态加载和执行。

- 如果目标是使用 LLVM 后端（如 x86、ARM、NVPTX 和 AMDGPU），代码生成主要由定义在 `src/codegen/llvm/codegen_llvm.cc` 中的 `CodeGenLLVM` 类完成。
- `CodeGenLLVM` 的作用是将 TVM 的 IR 转换为 LLVM 的 IR。这一步是重要的，因为 LLVM 提供了强大的优化和代码生成能力。

- 在生成 LLVM IR 后，`CodeGenLLVM` 会执行一些 LLVM 优化。这些优化可以提高生成代码的性能，利用 LLVM 的优化工具链来提升最终机器代码的执行效率。
- 最后，`CodeGenLLVM` 会生成适用于特定目标架构的机器代码，使得该代码可以在不同的硬件上高效运行。



实现了一个使用tvm库进行矩阵乘法的 CUDA 程序。该程序在设备上执行矩阵乘法运算，并测量其性能。

- 包含必要的库和头文件，包括 CUDA 运行时库和辅助函数
- 定义矩阵乘法的维度: 设置矩阵 \(A\) 的大小为 \(320* 640\)，矩阵 \(B\) 的大小为 \(640* 320\)。
- 构建计算图:使用 `te.placeholder` 定义输入矩阵 \(A\) 和 \(B\)。使用 `te.compute` 定义输出矩阵 \(C\) 的计算逻辑，利用 `te.sum` 进行矩阵乘法。
- 创建调度：使用 `te.create_schedule` 创建调度，并为 GPU 设置线程和块的调度。使用 `s[C].split` 和 `s[C].bind` 将计算任务分配到不同的 GPU 线程和块。
- 构建和运行函数 `build_and_run`：编译计算图为可执行的函数，并为输入矩阵分配随机数据。在设备上分配内存，创建 TVM 数组。计算 FLOPs，并在循环中执行矩阵乘法多次以计时。
- 计算性能指标:计算总运行时间和每秒浮点运算次数 (GFLOPS)，并输出结果。
- 执行代码: 调用 `build_and_run` 函数在 GPU 上执行矩阵乘法，并打印计算图的简化模式。

代码：

```python
import tvm
from tvm import te
import numpy as np
import time

# 定义矩阵乘法的大小
M = 320
N = 640
K = 320

# 定义矩阵乘法
A = te.placeholder((M, N), name='A')
B = te.placeholder((N, K), name='B')
k = te.reduce_axis((0, N), name='k')
C = te.compute((M, K), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

# 创建调度
s = te.create_schedule(C.op)

# GPU 线程调度
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

# 为 GPU 添加块和线程的调度
bx, tx = s[C].split(C.op.axis[0], factor=32)
by, ty = s[C].split(C.op.axis[1], factor=32)
s[C].bind(bx, block_x)
s[C].bind(by, block_y)
s[C].bind(tx, thread_x)
s[C].bind(ty, thread_y)

# 定义函数
def build_and_run(target_device="cuda", num_repeats=300):
    # 编译
    target = tvm.target.Target(target_device)
    f = tvm.build(s, [A, B, C], target=target, name='matmul')

    # 创建输入数据
    a_np = np.random.uniform(-1, 1, size=(M, N)).astype(np.float32)
    b_np = np.random.uniform(-1, 1, size=(N, K)).astype(np.float32)
    c_np = np.zeros((M, K), dtype=np.float32)

    # 在设备上分配内存
    dev = tvm.device(target_device, 0)
    a_tvm = tvm.nd.array(a_np, dev)
    b_tvm = tvm.nd.array(b_np, dev)
    c_tvm = tvm.nd.array(c_np, dev)

    # 计算 FLOPs（2 * M * N * K）
    flops = 2 * M * N * K
    
    # 运行并计时
    start_time = time.time()
    for i in range(num_repeats):
        f(a_tvm, b_tvm, c_tvm)
    dev.sync()  # 保证所有计算都已完成
    end_time = time.time()

    # 计算总时间和 GFLOPS
    total_time = end_time - start_time
    gflops = (flops * num_repeats) / (total_time * 1e9)

    # 输出结果
    print(f"Execution on {target_device} completed in {total_time:.4f} seconds for {num_repeats} iterations.")
    print(f"FLOPs: {flops} per matrix multiplication")
    print(f"GFLOPS: {gflops:.2f} GFLOPS")

# 在 GPU 上执行
build_and_run(target_device="cuda")
```

结果：

```
Execution on cuda completed in 0.1786 seconds for 300 iterations.
FLOPs: 131072000 per matrix multiplication
GFLOPS: 220.18 GFLOPS
```

实现了一个使用 TVM 的Auto-scheduling 进行算子优化。

- 定义一个带有偏置加法的矩阵乘法。这里使用了 TVM 张量表达式语言中的标准操作。区别在于函数定义上方使用了 `register_workload` 装饰器。该函数应返回输入/输出张量列表。通过这些张量，auto-scheduler 可以得到整个计算图。
- 定义函数后，可以为 auto_scheduler 创建要搜索的任务。为这个矩阵乘法指定了特定的参数，如这里是两个大小为 1024x1024 的矩阵乘法。然后创建一个 N=L=M=1024 和 dtype="float32" 的搜索任务
- `num_measure_trials` 表示搜索过程中可用的测试试验次数。用 `RecordToFile` 将测试记录记录到文件 `matmul.json` 中。测试记录可用于查询历史最佳、恢复搜索以及以后进行更多分析。
- auto-scheduling 完成后，可将 schedule 降级来查看 IR。auto-scheduler 执行合适的优化，包括多级循环切分、布局转换、并行化、向量化、循环展开和算子融合。

代码：

```python
import logging
import sys
import numpy as np
import tvm
from tvm import te
import tvm.testing

from tvm import autotvm
@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]
target = tvm.target.Target("llvm")
N = L = M = 1024
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)

# 检查计算图
print("Computational DAG:")
print(task.compute_dag)
log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
# 运行 auto-tuning（搜索）
task.tune(tune_option)
# 应用最佳 schedule
sch, args = task.apply_best(log_file)
print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
```

结果：

```
Computational DAG:
A = PLACEHOLDER [1024, 1024]
B = PLACEHOLDER [1024, 1024]
matmul(i, j) += (A[i, k]*B[k, j])
C = PLACEHOLDER [1024, 1024]
out(i, j) = (matmul[i, j] + C[i, j])
Lowered TIR:
@main = primfn(A_1: handle, B_1: handle, C_1: handle, out_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [1048576], []),
             B: Buffer(B_2: Pointer(float32), float32, [1048576], []),
             C: Buffer(C_2: Pointer(float32), float32, [1048576], []),
             out: Buffer(out_2: Pointer(float32), float32, [1048576], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C, out_1: out}
  preflattened_buffer_map = {A_1: A_3: Buffer(A_2, float32, [1024, 1024], []), B_1: B_3: Buffer(B_2, float32, [1024, 1024], []), C_1: C_3: Buffer(C_2, float32, [1024, 1024], []), out_1: out_3: Buffer(out_2, float32, [1024, 1024], [])} {
  allocate(auto_scheduler_layout_transform: Pointer(global float32), float32, [1048576]), storage_scope = global {
    for (ax0.ax1.fused.ax2.fused: int32, 0, 128) "parallel" {
      for (ax4: int32, 0, 256) {
        for (ax6: int32, 0, 4) {
          for (ax7: int32, 0, 8) {
            auto_scheduler_layout_transform_1: Buffer(auto_scheduler_layout_transform, float32, [1048576], [])[((((ax0.ax1.fused.ax2.fused*8192) + (ax4*32)) + (ax6*8)) + ax7)] = B[((((ax4*4096) + (ax6*1024)) + (ax0.ax1.fused.ax2.fused*8)) + ax7)]
          }
        }
      }
    }
    for (i.outer.outer.j.outer.outer.fused: int32, 0, 16384) "parallel" {
      allocate(matmul: Pointer(global float32x8), float32x8, [4]), storage_scope = global;
      for (i.outer.inner: int32, 0, 2) {
        matmul_1: Buffer(matmul, float32x8, [4], [])[0] = broadcast(0f32, 8)
        matmul_1[1] = broadcast(0f32, 8)
        matmul_1[2] = broadcast(0f32, 8)
        matmul_1[3] = broadcast(0f32, 8)
        for (k.outer: int32, 0, 256) {
          for (k.inner: int32, 0, 4) {
            let cse_var_2: int32 = (((floormod(i.outer.outer.j.outer.outer.fused, 128)*8192) + (k.outer*32)) + (k.inner*8))
            let cse_var_1: int32 = ((((floordiv(i.outer.outer.j.outer.outer.fused, 128)*8192) + (i.outer.inner*4096)) + (k.outer*4)) + k.inner)
             {
              matmul_1[0] = (matmul_1[0] + (broadcast(A[cse_var_1], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[1] = (matmul_1[1] + (broadcast(A[(cse_var_1 + 1024)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[2] = (matmul_1[2] + (broadcast(A[(cse_var_1 + 2048)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
              matmul_1[3] = (matmul_1[3] + (broadcast(A[(cse_var_1 + 3072)], 8)*auto_scheduler_layout_transform_1[ramp(cse_var_2, 1, 8)]))
            }
          }
        }
        for (i.inner: int32, 0, 4) {
          let cse_var_3: int32 = ((((floordiv(i.outer.outer.j.outer.outer.fused, 128)*8192) + (i.outer.inner*4096)) + (i.inner*1024)) + (floormod(i.outer.outer.j.outer.outer.fused, 128)*8))
          out[ramp(cse_var_3, 1, 8)] = (matmul_1[i.inner] + C[ramp(cse_var_3, 1, 8)])
        }
      }
    }
  }
}
```

实现了在 Relay 中定义神经网络，并为装有 TVM 的 NVIDIA GPU 生成 runtime 库。

- 使用 Relay 框架定义了 ResNet18 神经网络模型，设定批量大小为 1，图像形状为 (3, 224, 224)，输出类别数为 1000。
- 输出 ResNet18 模型的计算图结构，`show_meta_data=False` 表示不显示元数据。
- 设置优化级别为 3（包括算子融合、预计算、布局变换等优化），并指定 CUDA 作为目标设备，编译生成可在 GPU 上执行的库。
- 随机生成形状为 `(1, 3, 224, 224)` 的输入数据。创建一个执行模块，并将输入数据设置到模型中，然后运行模型并获取输出结果。输出结果中的前 10 个元素。
- 使用 TVM 的 `utils.tempdir` 创建临时目录，并将编译后的计算图、库和参数保存为文件，以便于后续部署时使用。
- 从保存的文件中加载编译模块，并使用相同的输入数据进行推理，获取输出结果。再次输出推理结果的前 10 个元素。
- 使用 `tvm.testing.assert_allclose` 检查重新加载的模块输出与最初输出是否一致，容差设置为 1e-5。

```python
import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape
)

# 想显示元数据则设置 show_meta_data=True
#print(mod.astext(show_meta_data=False))
# 为 NVIDIA GPU 编译
opt_level = 3
target = tvm.target.cuda()
with tvm.transform.PassContext(opt_level=opt_level):
    lib = relay.build(mod, target, params=params)
    
#创建图执行器，然后在 NVIDIA GPU 上运行该模块
# create random input
dev = tvm.cuda()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
module = graph_executor.GraphModule(lib["default"](dev))
# set input and parameters
module.set_input("data", data)
# run
module.run()
# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print(out.flatten()[0:10])
```

结果：

```
[0.00089283 0.00103331 0.0009094  0.00102275 0.00108751 0.00106737
 0.00106262 0.00095838 0.00110792 0.00113151]
```

```python
# create random input 图执行器，然后在 NVIDIA GPU 上运行该模块

dev = tvm.cuda()
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")

# create module
module = graph_executor.GraphModule(lib["default"](dev))

# set input and parameters
module.set_input("data", data)

# run
module.run()

# get output
out = module.get_output(0, tvm.nd.empty(out_shape)).numpy()

# Print first 10 elements of output
print(out.flatten()[0:10])
```

结果：

```
[0.00089283 0.00103331 0.0009094  0.00102275 0.00108751 0.00106737
 0.00106262 0.00095838 0.00110792 0.00113151]
```

```python
# 保存和加载编译模块 分别将计算图、库和参数保存到不同文件

from tvm.contrib import utils

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)
print(temp.listdir())

# 重新加载模块
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)
module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()

# 打印输出的前十个元素
print(out_deploy.flatten()[0:10])

# 检查来自部署模块的输出和原始输出是否一致
tvm.testing.assert_allclose(out_deploy, out, atol=1e-5)
```

结果：

```
['deploy_lib.tar']

[0.00089283 0.00103331 0.0009094  0.00102275 0.00108751 0.00106737
 0.00106262 0.00095838 0.00110792 0.00113151]
```

实现了将 ONNX 模型编译到 TVM Runtime并使用 TVMC 运行来自编译模块的模型

- 从指定的 URL 下载图像，并保存为 `imagenet_cat.png`。
- 使用 PIL 库将下载的图像大小调整为 224x224，以适应标准的图像输入要求（例如 ResNet）。
- 将图像数据从 HWC（Height-Width-Channel）格式转换为 NCHW（Channel-Height-Width）格式，这是 ONNX 模型的输入格式要求。
- 根据 ImageNet 的标准化方法，对图像进行归一化处理，减去均值 `imagenet_mean` 并除以标准差 `imagenet_stddev`。
- 将图像数据扩展一个维度，以符合神经网络模型所需的 batch 大小格式 (batch, channel, height, width)。
- 最终将预处理后的图像数据保存为 `imagenet_cat.npz`，用于后续推理。
- 从指定的 URL 下载 ImageNet 的类别标签列表，并保存为 `synset.txt`。
- 从保存的 `predictions.npz` 文件中加载输出张量，该文件应是神经网络推理后的结果。
- 使用 softmax 函数将模型的输出转化为概率分布。根据概率分数对输出进行排序，选出排名前 5 的类别，并打印它们的标签及对应的概率。

```
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np

img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# ONNX 需要 NCHW 输入, 因此对数组进行转换
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 进行标准化
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")
for i in range(img_data.shape[0]):
      norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

# 添加 batch 维度
img_data = np.expand_dims(norm_img_data, axis=0)

# 保存为 .npz（输出 imagenet_cat.npz）
np.savez("imagenet_cat", data=img_data)

import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# 打开并读入输出张量
if os.path.exists(output_file):
    with np.load(output_file) as data:
        scores = softmax(data["output_0"])
        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
```

结果：

```
class='n02123045 tabby, tabby cat' with probability=0.621104
class='n02123159 tiger cat' with probability=0.356378
class='n02124075 Egyptian cat' with probability=0.019712
class='n02129604 tiger, Panthera tigris' with probability=0.001215
class='n04040759 radiator' with probability=0.000262
```

实现了使用 AutoTVM 在 TVM 中编译和优化 ONNX 模型。

- 使用 `onnx.load()` 加载 ONNX 模型。

- 下载一张图像并将其调整为 224x224 像素，这是 ResNet 等模型的标准输入大小。根据 ImageNet 的标准对图像进行归一化，并调整为 NCHW 格式。

- 使用 Relay 前端编译模型，并指定目标架构（CUDA 用于 GPU）。

- 构建模型并将其转换为图模块以便执行。

- 使用 TVM 的运行时运行模型以获取预测结果，并使用 softmax 处理结果以获得每个类别的概率。

- 使用 `timeit` 测量推理运行时间，并保存优化和未优化模型的结果。

- 使用 TVM 的 AutoTVM 中的 `XGBTuner` 启动调优过程。

- 设置调优选项并在从模型中提取的任务上运行调优。

- 在调优后，使用在调优过程中找到的最佳配置重新构建模型，并验证优化模型的预测结果。

- 打印优化模型和未优化模型的性能指标以进行比较。

  ```python
  import onnx
  from tvm.contrib.download import download_testdata
  from PIL import Image
  import numpy as np
  import tvm.relay as relay
  import tvm
  from tvm.contrib import graph_executor
  
  model_url = (
      "https://github.com/onnx/models/raw/main/"
      "vision/classification/resnet/model/"
      "resnet50-v2-7.onnx"
  )
  
  model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
  onnx_model = onnx.load(model_path)
  
  img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
  img_path = download_testdata(img_url, "imagenet_cat.png", module="data")
  
  # 重设大小为 224x224
  
  resized_image = Image.open(img_path).resize((224, 224))
  img_data = np.asarray(resized_image).astype("float32")
  
  # 输入图像是 HWC 布局，而 ONNX 需要 CHW 输入，所以转换数组
  
  img_data = np.transpose(img_data, (2, 0, 1))
  
  # 根据 ImageNet 输入规范进行归一化
  
  imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
  imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
  norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev
  
  # 添加 batch 维度，期望 4 维输入：NCHW。
  
  img_data = np.expand_dims(norm_img_data, axis=0)
  
  # 为 numpy 的 RNG 设置 seed，得到一致的结果
  
  np.random.seed(0)
  
  target = "cuda"
  # 可用 Netron 工具检查输入名称
  input_name = "data"
  shape_dict = {input_name: img_data.shape}
  
  mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
  
  with tvm.transform.PassContext(opt_level=3):
      lib = relay.build(mod, target=target, params=params)
  
  dev = tvm.device(str(target), 0)
  module = graph_executor.GraphModule(lib["default"](dev))
  
  #在 TVM Runtime 执行
  dtype = "float32"
  module.set_input(input_name, img_data)
  module.run()
  output_shape = (1, 1000)
  tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
  
  #收集基本性能数据
  import timeit
  timing_number = 10
  timing_repeat = 10
  unoptimized = (
      np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
      * 1000
      / timing_number
  )
  unoptimized = {
      "mean": np.mean(unoptimized),
      "median": np.median(unoptimized),
      "std": np.std(unoptimized),
  }
  print(unoptimized)
  ```

  结果：

  ```
  class='n02123045 tabby, tabby cat' with probability=0.621103
  class='n02123159 tiger cat' with probability=0.356379
  class='n02124075 Egyptian cat' with probability=0.019712
  class='n02129604 tiger, Panthera tigris' with probability=0.001215
  class='n04040759 radiator' with probability=0.000262
  ```

  ```python
  #调优模型
  import tvm.auto_scheduler as auto_scheduler
  from tvm.autotvm.tuner import XGBTuner
  from tvm import autotvm
  
  logging.basicConfig(level=logging.DEBUG)
  
  number = 10
  repeat = 1
  min_repeat_ms = 100  # 对于 GPU 设置为一个合理值，通常不为 0
  timeout = 10  # 秒
  
  # 创建 TVM 运行器，针对 GPU 不需要 CPU 缓存刷新
  runner = autotvm.LocalRunner(
      number=number,
      repeat=repeat,
      timeout=timeout,
      min_repeat_ms=min_repeat_ms,
      enable_cpu_cache_flush=False,  # GPU 不需要清空 CPU 缓存
  )
  
  # 使用 XGBoost 算法来指导搜索。对于 GPU 推荐 3000-4000 次试验
  tuning_option = {
      "tuner": "xgb",
      "trials": 4000,  # 对于 GPU 调优，推荐更高的试验次数
      "early_stopping": 800,  # 设置一个较大的早停值
      "measure_option": autotvm.measure_option(
          builder=autotvm.LocalBuilder(build_func="default"), 
          runner=runner
      ),
      "tuning_records": "resnet-50-v2-autotuning-gpu.json",  # 记录调优结果的文件
  }
  
  # 设置目标为 CUDA，表示 GPU
  target = "cuda"
  
  # 从 onnx 模型中提取任务
  tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
  
  # 按顺序调优提取的任务
  for i, task in enumerate(tasks):
      prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
  
      # 选择 XGBoost 调优器
      tuner = "xgb"
  
      # 创建调优器
      if tuner == "xgb":
          tuner_obj = XGBTuner(task, loss_type="reg")
      else:
          raise ValueError("Invalid tuner: " + tuner)
  
      # 开始调优
      tuner_obj.tune(
          n_trial=min(tuning_option["trials"], len(task.config_space)),
          early_stopping=tuning_option["early_stopping"],
          measure_option=tuning_option["measure_option"],
          callbacks=[
              autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
              autotvm.callback.log_to_file(tuning_option["tuning_records"]),
          ],
      )
  ```

  结果：

  ```
  [Task 25/25]  Current/Best:    0.00/   0.00 GFLOPS | Progress: (0/20) | 0.00 s
  [Task 25/25]  Current/Best:    1.56/   2.93 GFLOPS | Progress: (4/20) | 9.63 s
  [Task 25/25]  Current/Best:    5.65/   7.64 GFLOPS | Progress: (8/20) | 18.43 s
  [Task 25/25]  Current/Best:    5.95/   7.64 GFLOPS | Progress: (12/20) | 29.31 s
  [Task 25/25]  Current/Best:    5.80/   9.36 GFLOPS | Progress: (16/20) | 36.11 s
  [Task 25/25]  Current/Best:    2.94/   9.36 GFLOPS | Progress: (20/20) | 51.33 s
  ```

  ```python
  #使用调优数据编译优化模型，获取存储在 resnet-50-v2-autotuning.json（上述调优过程的输出文件）中的调优记录
  with autotvm.apply_history_best(tuning_option["tuning_records"]):
      with tvm.transform.PassContext(opt_level=3, config={}):
          lib = relay.build(mod, target=target, params=params)
  
  dev = tvm.device(str(target), 0)
  module = graph_executor.GraphModule(lib["default"](dev))
  
  #验证优化模型是否运行并产生相同的结果：
  dtype = "float32"
  module.set_input(input_name, img_data)
  module.run()
  output_shape = (1, 1000)
  tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()
  
  scores = softmax(tvm_output)
  scores = np.squeeze(scores)
  ranks = np.argsort(scores)[::-1]
  for rank in ranks[0:5]:
      print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
  ```

  结果：

  ```
  class='n02123045 tabby, tabby cat' with probability=0.621104
  class='n02123159 tiger cat' with probability=0.356378
  class='n02124075 Egyptian cat' with probability=0.019712
  class='n02129604 tiger, Panthera tigris' with probability=0.001215
  class='n04040759 radiator' with probability=0.000262
  ```

  ```python
  #比较调优和未调优的模型，收集与此优化模型相关的一些基本性能数据，并将其与未优化模型进行比较。
  import timeit
  
  timing_number = 10
  timing_repeat = 10
  optimized = (
      np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
      * 1000
      / timing_number
  )
  optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}
  
  print("optimized: %s" % (optimized))
  print("unoptimized: %s" % (unoptimized))
  ```

  结果：

  ```
  optimized: {'mean': 407.31687583000166, 'median': 407.3377107500164, 'std': 1.692177042688564}
  unoptimized: {'mean': 495.13895513002353, 'median': 494.6680843500417, 'std': 1.3081147373726523}
  ```

### 3.6 OpenXLA (NVIDIA)

![OpenXLA 生态系统](https://github.com/openxla/xla/raw/main/docs/images/openxla.svg)

此图为官图需要重绘

OpenXLA 是一个为多种硬件设备加速深度学习和机器学习模型执行的开放框架。它的设计目的是在不同硬件平台（如GPU、TPU、CPU和加速卡）上优化机器学习工作负载。OpenXLA 由多个子组件组成，这些组件为不同层次的优化和执行提供支持。

#### 3.5.1 技术栈架构

![OpenXLA 生态系统](https://github.com/openxla/xla/raw/main/docs/images/openxla.svg)

此图为官图需要重绘

OpenXLA 是一个为多种硬件设备加速深度学习和机器学习模型执行的开放框架。它的设计目的是在不同硬件平台（如GPU、TPU、CPU和加速卡）上优化机器学习工作负载。OpenXLA 由多个子组件组成，这些组件为不同层次的优化和执行提供支持。

#### 3.6.2 运行时环境层

OpenXLA 可以通过底层库（例如 CUDA Runtime 或 CUDA Driver API）与 GPU 交互，但它不是直接用于设备查询或管理的工具。OpenXLA 的主要作用是为机器学习模型提供跨硬件的优化执行支持。OpenXLA 依赖于 CUDA API 进行设备信息查询。

- 定义了一个宏 `CHECK_CUDA`，用于检查 CUDA API 调用是否成功。如果失败，获取错误信息并退出程序。
- 调用 `cuInit(0)` 初始化 CUDA 驱动程序。必须在所有 CUDA API 调用之前执行。
- 使用 `cuDeviceGetCount(&deviceCount)` 获取系统中可用的 CUDA 设备数量，并打印出来。
- 使用 `cuDeviceGet(&device, i)` 获取每个 CUDA 设备的句柄，用于后续查询设备信息。
- 使用 `cuDeviceGetName(name, sizeof(name), device)` 获取每个设备的名称（例如 GPU 型号）。
- 使用 `cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)` 和 `cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)` 获取设备的计算能力主版本和次版本。
- 使用 `cuDeviceTotalMem(&totalMem, device)` 获取设备的总内存大小（以字节为单位），并转换为 MB 打印出来。

```c++
#include <stdio.h>
#include <cuda.h>

// CUDA 错误检查宏
#define CHECK_CUDA(call) do { \
    CUresult result = call; \
    if (result != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(result, &errStr); \
        printf("CUDA Error: %s\n", errStr); \
        return -1; \
    } \
} while (0)

int main() {
    // 初始化 CUDA Driver API
    CHECK_CUDA(cuInit(0));

    // 获取设备数量
    int deviceCount = 0;
    CHECK_CUDA(cuDeviceGetCount(&deviceCount));

    printf("CUDA 设备数量: %d\n", deviceCount);

    // 遍历每个设备，获取设备信息
    for (int i = 0; i < deviceCount; ++i) {
        CUdevice device;
        char name[128];
        int major = 0, minor = 0;

        // 获取设备句柄
        CHECK_CUDA(cuDeviceGet(&device, i));

        // 获取设备名称
        CHECK_CUDA(cuDeviceGetName(name, sizeof(name), device));

        // 获取设备的计算能力 (Compute Capability)
        CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

        // 获取设备的总内存
        size_t totalMem = 0;
        CHECK_CUDA(cuDeviceTotalMem(&totalMem, device));

        printf("设备 %d: %s\n", i, name);
        printf("  计算能力: %d.%d\n", major, minor);
        printf("  总内存: %zu MB\n", totalMem / (1024 * 1024));
    }

    return 0;
}

```

结果：

```
CUDA 设备数量: 1
设备 0: NVIDIA GeForce RTX 4080 SUPER
  计算能力: 8.9
  总内存: 16072 MB
```

#### 3.6.3 计算库层

在 XLA（Accelerated Linear Algebra）中使用自定义调用（Custom Call）机制，结合 XLA FFI（外部函数接口，Foreign Function Interface）来实现用户定义的操作。使用自定义调用在 CPU 上计算 `A[i] = B[i % 128]+ C[i]`。

- `xla::XlaBuilder`：XLA 提供的用于构建计算图的类，这里实例化了一个名为 "do_it" 的构建器 `b`。
- `xla::Parameter`：定义两个输入参数 `param0` 和 `param1`。其中 `param0` 是一个长度为 128 的 1D 浮点型（F32）数组，`param1` 是长度为 2048 的 1D 浮点型数组。
- `xla::CustomCall`：这是 XLA 中执行自定义操作的关键调用。通过传递 `"do_custom_call"` 字符串来指定自定义调用的名称，表示需要调用一个外部定义的函数。该自定义操作接收两个输入（`param0` 和 `param1`），输出结果的形状是一个长度为 2048 的 F32 数组。
- `BufferF32`：这是 XLA FFI 中的类型别名，表示一个 1D 的浮点型（F32）缓冲区。
- in0` 和 `in1` 是输入缓冲区（分别为 param0 和 param1 的数据），它们的数据类型为 `BufferF32`out` 是输出缓冲区，存储结果。该函数的逻辑为：将 `in0` 和 `in1` 中的数据进行逐元素相加，并将结果写入输出缓冲区。注意这里通过 `i % d0` 来处理 `in0`，使得其在计算时按顺序重复。`assert` 检查输出缓冲区的维度，确保与 `in1` 的维度相同。
- 定义了一个处理器 `handler`，并将它绑定到 `do_custom_call` 函数上。通过这种绑定，FFI 可以知道自定义调用应该如何匹配到 C++ 函数。绑定过程中明确指定了函数的参数类型和返回值类型为 `Buffer`（即 1D 缓冲区）。
- 将处理器 `handler` 注册到 XLA FFI，表示它将在 "Host" 平台上可用。
- `"do_custom_call"` 是自定义调用的名称，与 `xla::CustomCall` 中使用的名称一致。
- `xla::ffi::GetXlaFfiApi()` 获取当前的 XLA FFI API 实例，确保处理器能够正确注册到 XLA。

```c++
#include "xla/client/xla_builder.h"
#include "xla/service/custom_call_target_registry.h"

void do_it() {
  xla::XlaBuilder b("do_it");
  xla::XlaOp param0 =
      xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
  xla::XlaOp param1 =
      xla::Parameter(&b, 1, xla::ShapeUtil::MakeShape(xla::F32, {2048}), "p1");
  xla::XlaOp custom_call =
      xla::CustomCall(&b, "do_custom_call", /*operands=*/{param0, param1},
        /*shape=*/xla::ShapeUtil::MakeShape(xla::F32, {2048}),
        /*opaque=*/"", /*has_side_effect=*/false,
        /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
        /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
        /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
}

// Constrain custom call arguments to rank-1 buffers of F32 data type.
using BufferF32 = xla::ffi::BufferR1<xla::ffi::DataType::F32>;

// Implement a custom call as a C+ function. Note that we can use `Buffer` type
// defined by XLA FFI that gives us access to buffer data type and shape.
xla::ffi::Error do_custom_call(BufferF32 in0, BufferF32 in1,
                               xla::ffi::Result<BufferF32> out) {
  size_t d0 = in0.dimensions[0];
  size_t d1 = in1.dimensions[0];

  // Check that dimensions are compatible.
  assert(out->dimensions[0] == d1 && "unexpected dimensions");

  for (size_t i = 0; i < d1; ++i) {
    out->data[i] = in0.data[i % d0] + in1.data[i];
  }
}

// Explicitly define an XLA FFI handler signature and bind it to the
// `do_custom_call` implementation. XLA FFI handler can automatically infer
// type signature from the custom call function, but it relies on magical
// template metaprogramming an explicit binding provides and extra level of
// type checking and clearly states custom call author intentions.
XLA_FFI_DEFINE_HANDLER(handler, do_custom_call,
                       ffi::Ffi::Bind()
                           .Arg<Buffer>()
                           .Arg<Buffer>()
                           .Ret<Buffer>());

// Registers `handler` with and XLA FFI on a "Host" platform.
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "do_custom_call",
                         "Host", handler);
```

在原有的 XLA 的自定义调用实现上进行了扩展，增加了 GPU 加速部分，主要通过 CUDA 来并行处理自定义操作的逻辑，计算 `A[i] = B[i % 128] + C[i]`。

- 构建了 XLA 的计算图，通过 `xla::CustomCall` 调用了名为 `"do_custom_call"` 的自定义操作。它定义了两个输入参数 `param0` 和 `param1`，并设置输出为长度为 2048 的浮点数数组。
- `const float* in0, const float* in1, float* out`：输入 `in0` 和 `in1` 是常量浮点型数组指针，`out` 是输出数组指针。`size_t idx = blockIdx.x * blockDim.x + threadIdx.x`：计算当前线程的全局索引 `idx`。`blockIdx.x` 是当前线程块的索引，`blockDim.x` 是每个线程块的大小，`threadIdx.x` 是当前线程在块内的索引。`out[idx] = in0[idx % 128] + in1[idx]`：对于每个线程，执行 `in0[idx % 128] + in1[idx]`，并将结果写入 `out[idx]`。`in0` 的大小为 128，因此使用 `% 128` 使得 `in0` 的数据循环重复使用，而 `in1` 和 `out` 都是长度为 2048。
- `block_dim` 和 `grid_dim`：用于定义 CUDA kernel 的执行配置。`block_dim` 设置为 64，表示每个线程块中有 64 个线程，`grid_dim` 设置为 `2048 / 64`，即 32 个线程块。每个线程块并行处理 64 个数据元素，共 2048 个数据元素。
- `custom_call_kernel<<<grid_dim, block_dim, 0, stream>>>(in0.data, in1.data, out->data)`：通过 CUDA 启动 `custom_call_kernel` 内核，传入输入和输出数据指针，以及 CUDA 流 `stream`，让 GPU 并行执行数据计算。
- `XLA_FFI_DEFINE_HANDLER`：定义一个新的 XLA FFI 处理器 `handler`，并将其绑定到 `do_custom_call` 函数。
- `.Ctx<xla::ffi::PlatformStream<CUstream>>()`：这行代码表明 `do_custom_call` 函数需要接受一个 CUDA 流 `CUstream` 作为上下文，以便在 GPU 上执行自定义调用。
- `.Arg<BufferF32>()`：定义两个参数，类型为 `BufferF32`（浮点数组）。`.Ret<BufferF32>()`：定义返回值为 `BufferF32`。
- `XLA_FFI_REGISTER_HANDLER`：将定义好的 `handler` 注册到 XLA FFI 中，使得 XLA 可以识别并调用这个自定义操作。

```c++
void do_it() { /* same implementation as above */ }

__global__ custom_call_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, BufferF32 in0, BufferF32 in1,
                    xla::ffi::Result<BufferF32> out) {
  size_t d0 = in0.dimensions[0];
  size_t d1 = in1.dimensions[0];
  size_t d2 = out->dimensions[0];

  assert(d0 == 128 && d1 == 2048 && d2 == 2048 && "unexpected dimensions");

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_call_kernel<<<grid_dim, block_dim, 0, stream>>>(
    in0.data, in1.data, out->data);
}

XLA_FFI_DEFINE_HANDLER(handler, do_custom_call,
                       ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<CUstream>>()
                           .Arg<BufferF32>()
                           .Arg<BufferF32>()
                           .Ret<BufferF32>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "do_custom_call",
                         "CUDA", handler);
```

为 TensorFlow 启用 XLA，使用@tf.function(jit_compile=True)进行显式编译，显式编译 API 提供精细的控制，用于选择应编译哪些函数。例如，以下执行 MNIST 训练的 TensorFlow 函数使用 XLA 进行编译：

```
@tf.function(jit_compile=True)
def train_mnist(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
      predicted_labels = layer(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))
    layer_variables = layer.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))
```

`tfcompile` 是 XLA编译器工具，可以将 TensorFlow 图进行提前（AOT）编译，生成可执行代码。它有助于减少二进制文件的整体大小，并能避免部分运行时开销。`tfcompile` 接收一个子图（通过 TensorFlow 的 Feed 和 Fetch 概念定义），并生成实现该子图的函数。Feed 对应函数的输入参数，Fetch 对应函数的输出参数。所有输入必须通过 Feed 完全指定，生成的子图不能包含占位符或变量节点。通常的做法是将所有占位符和变量标记为 Feed，以确保最终生成的子图中没有这些节点。

使用 tfcompile 编译 TensorFlow 子图，首先，需要定义一个简单的 TensorFlow 模型或子图。以下是一个定义子图的示例，输入为标量，输出为其平方。

```
import tensorflow as tf

# 创建计算图
def simple_graph(x):
    return tf.math.square(x)

# 输入符号化
x = tf.placeholder(dtype=tf.float32, shape=(), name='input')

# 定义子图
y = simple_graph(x)

# 将计算图保存到文件
with tf.Session() as sess:
    tf.io.write_graph(sess.graph_def, './', 'simple_graph.pbtxt')
```

`tfcompile` 需要一个配置文件，指定输入、输出及其他信息。配置文件 `config.pbtxt` 示例：

```
# config.pbtxt
feed {
  id { node_name: "input" }
  shape { dim { size: 1 } }  # 指定输入张量的形状
}
fetch {
  id { node_name: "Square" }  # 这是子图输出节点的名称
}
```

使用 `tfcompile` 编译器编译生成可执行二进制文件。生成的 `.o` 文件还需要链接到可执行程序。下面是 C++ 示例，展示如何使用生成的二进制文件：

```c++
#include <iostream>
#include "compiled_graph.o"

int main() {
    // 创建输入张量
    MyCompiledGraph computation;
    float input_value = 3.0;
    float output_value;

    // 执行计算
    computation.compute(&input_value, &output_value);

    std::cout << "输入值: " << input_value << " 的平方是: " << output_value << std::endl;
    return 0;
}
```

编译运行后输出如下内容：

```
输入值: 3 的平方是: 9
```

为 pytorch启用 XLA，PyTorch/XLA 使用与常规 PyTorch 相同的接口，但有一些附加功能。导入会`torch_xla`初始化 PyTorch/XLA，并 `xm.xla_device()`返回当前 XLA 设备。

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)
```

结果

```
xla:0
tensor([[ 0.1028, -1.4783],
        [-0.4271,  1.3415]], device='xla:0')
```

与其他设备类型一样，XLA 张量仅与同一设备上的其他 XLA 张量配合使用。

```python
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20)
l_out = linear(l_in)
print(l_out)
# Input tensor is not an XLA tensor: torch.FloatTensor
```

张量从 CPU 移动到 XLA 设备：当张量从 CPU 移动到 XLA 设备（如 TPU、GPU）时，数据会被复制到目标设备的内存中。这意味着可以在加速硬件上执行计算。同样，XLA 设备上的张量可以移动回 CPU，在这个过程中，数据会从设备复制回 CPU 的内存。一旦张量数据被复制到另一台设备，两个设备上的张量副本之间不会有任何联系。每个副本在各自的设备内存中独立存在。

应在保存之前将 XLA 张量移至 CPU，如以下代码片段所示：

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()

t0 = torch.randn(2, 2, device=device)
t1 = torch.randn(2, 2, device=device)

tensors = (t0.cpu(), t1.cpu())

torch.save(tensors, 'tensors.pt')

tensors = torch.load('tensors.pt')

t0 = tensors[0].to(device)
t1 = tensors[1].to(device)
print(t0)
print(t1)
```

结果

```
tensor([[ 0.1028, -1.4783],
        [-0.4271,  1.3415]], device='xla:0')
tensor([[ 0.8257,  0.3266],
        [ 0.9146, -0.2747]], device='xla:0')
```

#### 3.6.4 编程模型和语言

使用了 `PyTorch XLA` 来在 XLA（如 TPU 等加速设备）上运行张量操作。

- 引入 `torch`、`torch_xla` 和 `torch_xla.core.xla_model`，用于在 XLA 设备上执行 PyTorch 操作。

- 使用 `torch.randn(2, 2, device=xm.xla_device())` 创建一个 2x2 的随机张量 `t`，并将其分配到 XLA 设备。

- 创建两个 2x2 的随机张量 `t0` 和 `t1`，并进行逐元素加法和矩阵乘法，打印结果。

- 创建一个大小为 10 的随机输入向量 `l_in`，并将其分配到 XLA 设备。
- 定义一个输入特征为 10、输出特征为 20 的线性层 `linear`，并迁移到 XLA 设备。
- 将输入 `l_in` 传入线性层，得到输出 `l_out`，并打印输出结果。

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm

t = torch.randn(2, 2, device=xm.xla_device())
print(t.device)
print(t)

t0 = torch.randn(2, 2, device=xm.xla_device())
t1 = torch.randn(2, 2, device=xm.xla_device())
print(t0 + t1)
print(t0.mm(t1))

#神经网络
l_in = torch.randn(10, device=xm.xla_device())
linear = torch.nn.Linear(10, 20).to(xm.xla_device())
l_out = linear(l_in)
print(l_out)
```

结果

```
xla:0
tensor([[ 0.1028, -1.4783],
        [-0.4271,  1.3415]], device='xla:0')
tensor([[ 1.7679,  0.2210],
        [ 0.5831, -1.5733]], device='xla:0')
tensor([[ 0.6698, -0.5113],
        [ 0.9527,  0.2601]], device='xla:0')
tensor([-0.8333,  0.4356,  0.4277, -0.3944,  0.8075,  0.3516,  0.0455,  0.0778,
        -0.0822,  0.4418, -0.7217,  0.3582, -0.7285,  0.1117, -0.0466, -0.7045,
        -0.1443,  0.3461, -0.3151, -0.6094], device='xla:0',
       grad_fn=<AddBackward0>)
```

实现了一个使用 `PyTorch XLA` 再 TPU 训练和评估 MNIST 手写数字分类模型的完整流程，包括数据加载、模型构建、训练、保存和推理。

- 引入所需的 PyTorch 和 Torch XLA 库，以及 MNIST 数据集和数据处理工具。设置设备为 TPU，使用 `xm.xla_device()`。
- 使用 `transforms.Compose` 创建数据转换，将 MNIST 数据集中的图像转换为张量。下载 MNIST 训练集并创建数据加载器 `train_loader`，设置批量大小为 64，并随机打乱数据。
- 定义一个简单的神经网络模型，包括：扁平化层，将 28x28 的图像展平成一维。128 单元的全连接层，使用 ReLU 激活函数。10 单元的全连接层，使用 LogSoftmax 激活函数。将模型迁移到 TPU 设备。
- 使用负对数似然损失函数 `NLLLoss`。使用随机梯度下降优化器 `SGD`，学习率为 0.01，动量为 0.9。
- 对训练数据进行迭代：清零优化器的梯度。将数据和目标迁移到 TPU 设备。通过模型进行前向传播，计算损失。进行反向传播以计算梯度。更新模型参数。调用 `xm.mark_step()` 同步 TPU。
- 使用 `torch.save()` 保存训练好的模型到 `mnist_model_full.pth` 文件中。
- 加载保存的模型，并将其迁移到 TPU 设备，切换到评估模式。
- 在不计算梯度的上下文中：遍历测试数据，迁移到 TPU 设备。进行前向传播，计算输出。使用 `torch.max()` 获取预测结果的最大值索引。打印预测结果，且仅处理一个批次作为示例。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

# 设备设定（TPU）
device = xm.xla_device()

# 数据集与数据加载器设定
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 模型设定
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
).to(device)

# 损失函数和优化器设定
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练循环
for data, target in train_loader:
    optimizer.zero_grad()
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    
    optimizer.step()
    xm.mark_step()  # TPU同步

# 保存整个模型
torch.save(model, 'mnist_model_full.pth')

# 模型推理
import torch

# 加载整个模型
model = torch.load('mnist_model_full.pth').to(device)
model.eval()  # 切换到评估模式

# 加载测试数据
test_dataset = MNIST(root='data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 使用模型进行推理
with torch.no_grad():  # 禁用梯度计算以加快推理
    for data, _ in test_loader:
        data = data.to(device)
        output = model(data)
        xm.mark_step()  # TPU同步
        
        # 获取预测结果
        _, predicted = torch.max(output, 1)
        print(predicted)
        break  # 仅处理一个批次的示例
```

结果

```
tensor([7, 2, 1, 0, 4, 1, 4, 9, 6, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4, 9, 6, 6, 5,
        4, 0, 7, 4, 0, 1, 3, 1, 3, 6, 7, 2, 7, 1, 2, 1, 1, 7, 4, 2, 3, 5, 1, 2,
        4, 4, 6, 3, 5, 5, 6, 0, 4, 1, 9, 5, 7, 8, 4, 3], device='xla:0')
```

将一个 PyTorch 模型导出并转换为一种适合跨平台应用的格式（ StableHLO ），以便进行优化、部署和进一步分析。

- 模型加载：加载了预训练的 ResNet-18 模型，使用 `torchvision` 提供的默认权重。
- 样本输入生成：创建了一个形状为 `(4, 3, 224, 224)` 的随机张量，模拟输入的图像数据。
- 模型导出：使用 `export` 函数将 ResNet-18 模型导出为中间表示，以便后续处理。
- 转换为 StableHLO：将导出的模型转换为 StableHLO 格式，适用于跨平台优化和部署。
- 输出 StableHLO 文本：打印模型前向计算图的 StableHLO 文本表示的前 400 个字符，以供检查和分析。

```python
import torch
import torchvision
from torch.export import export

resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
sample_input = (torch.randn(4, 3, 224, 224), )
exported = export(resnet18, sample_input)

from torch_xla.stablehlo import exported_program_to_stablehlo

stablehlo_program = exported_program_to_stablehlo(exported)
print(stablehlo_program.get_stablehlo_text('forward')[0:400],"\n...")
```

结果

```
module @IrToHlo.484 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1000xf32>, %arg1: tensor<1000x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512x256x1x1xf32>, %arg7: tensor<256xf32>, %arg8: tensor<256xf32>, %arg9: tensor<25 
...
```

- 定义一个简单的加法模型，并创建输入数据。
- 将模型导出为中间表示，并转换为 StableHLO 格式，便于跨平台应用和优化。
- 最后，输出转换后的模型信息，便于分析和调试。

```python
import torch
import torch.nn as nn
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

# 定义一个简单的加法模型
class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y

# 创建模型实例
add_model = AddModel()

# 创建示例输入
x_input = torch.randn(4, 3, 224, 224)  # 第一个输入
y_input = torch.randn(4, 3, 224, 224)  # 第二个输入

# 使用 export 函数导出模型
exported = export(add_model, (x_input, y_input))

# 将导出的模型转换为 StableHLO 格式
stablehlo_program = exported_program_to_stablehlo(exported)

# 打印 StableHLO 程序文本的一部分
print(stablehlo_program.get_stablehlo_text('forward')[0:400], "\n...")
```

结果

```
module @IrToHlo.8 attributes {mhlo.cross_program_prefetches = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<4x3x224x224xf32>, %arg1: tensor<4x3x224x224xf32>) -> tensor<4x3x224x224xf32> {
    %0 = stablehlo.add %arg1, %arg0 : tensor<4x3x224x224xf32>
    return %0 : tensor<4x3x224x224xf32>
  }
}
```

实现了使用 TensorFlow 定义一个简单的神经网络模型，生成随机输入，并使用 XLA（加速线性代数）优化进行前向传播。

- 使用 `tf.config.list_physical_devices('GPU')` 检查可用的 GPU 数量。输出可用 GPU 的数量。
- 使用 `tf.keras.Sequential` 创建一个顺序模型。第一层是一个全连接层（Dense），有 10 个单元，输入维度为 10，激活函数为 ReLU。第二层是另一个全连接层，包含 5 个单元，激活函数为 softmax。
- 定义批量大小（`batch_size`）为 16，输入向量维度（`input_vector_dim`）为 10。使用 `tf.random.normal` 生成形状为 `(16, 10)` 的随机输入。
- 使用 `@tf.function(jit_compile=True)` 装饰器定义前向传播函数，以启用 XLA 优化。函数接受输入并返回模型的输出。
- 调用前向传播函数 `forward_pass`，传入随机输入进行计算。

```python
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Define the model
model = tf.keras.Sequential(
    [tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"),
     tf.keras.layers.Dense(5, activation="softmax")]
)

# Generate random inputs for the model
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass
_ = model(random_inputs)

# Compile the model function with XLA optimization
@tf.function(jit_compile=True)
def forward_pass(inputs):
    return model(inputs)

# Run the forward pass with XLA
_ = forward_pass(random_inputs)

```

结果

```
I0000 00:00:1727407770.382644 1007512 service.cc:146] XLA service 0x8ec22c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
I0000 00:00:1727407770.382662 1007512 service.cc:154]   StreamExecutor device (0): NVIDIA GeForce RTX 4080 SUPER, Compute Capability 8.9
2024-09-27 11:29:30.387574: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2024-09-27 11:29:31.040309: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:531] Loaded cuDNN version 8907
I0000 00:00:1727407771.151882 1007512 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
```

### 3.7 OpenACC

#### 3.7.1 技术栈架构

OpenACC 是一种用于异构计算系统（如 GPU 加速器）的编程模型，它允许程序员通过指令简化代码的并行化和加速。它的技术栈可以分为几个层次，从底层硬件到高层的应用框架，包括系统软件层、运行时环境层、编程模型和语言层、计算库层和框架模型层。

##### 1. 系统软件层（System Software Layer）
**主要组件**：
- **操作系统（Operating System）**：操作系统是 OpenACC 应用程序运行的基础。通常，Linux 是高性能计算系统中最常用的操作系统，支持各类硬件加速器（GPU、FPGA等）的驱动。
- **驱动程序（Driver）**：如 CUDA 驱动（针对 NVIDIA GPU）或 ROCm 驱动（针对 AMD GPU），这些驱动为硬件提供低级访问接口，并支持高层编程模型（如 OpenACC）与硬件之间的通信。

##### 2. 运行时环境层（Runtime Environment Layer）
OpenACC 编译器会将程序转换为包含并行指令的代码，而运行时环境层则负责管理这些指令的执行，包括内存管理、数据移动和调度。

**主要组件**：
- **OpenACC Runtime Library**：OpenACC 运行时库支持运行时系统的指令调度和执行。它负责管理并行任务的启动、内存分配、主机与设备之间的数据传输等工作。
- **CUDA/ROCm Runtime**：如果 OpenACC 程序运行在 NVIDIA 或 AMD GPU 上，实际的并行执行由底层 CUDA 或 ROCm 运行时环境完成。

##### 3. 编程模型和语言层（Programming Model and Language Layer）
这是 OpenACC 核心的层次，程序员使用 OpenACC 的编程模型和语言构建并行程序。

**主要组件**：
- **OpenACC 规范**：OpenACC 使用编译指令（directives）的方式对现有代码进行注释，指示编译器如何并行化和加速代码。指令以 `#pragma acc` 开始，附带对并行执行、循环分配、数据传输等操作的具体说明。
- **C/C++ 和 Fortran**：OpenACC 编译指令可以与标准的 C/C++ 和 Fortran 语言配合使用，便于将现有代码改造为并行化代码。

##### 4. 计算库层（Compute Libraries Layer）
为了进一步提升性能和开发效率，OpenACC 编程环境下也可以使用许多预构建的高性能计算库。

**主要组件**：
- **cuBLAS、cuFFT（针对 NVIDIA）**：这些库为线性代数、傅里叶变换等常用计算提供高效实现，可以在 OpenACC 应用中被调用，从而减少手动编写复杂并行代码的需求。
- **rocBLAS、rocFFT（针对 AMD）**：这是 AMD 提供的类似库，支持基于 ROCm 的加速计算。
- **OpenACC 兼容的第三方库**：一些第三方库可以与 OpenACC 代码集成，处理专门的计算需求。

##### 5. 框架模型层（Framework Layer）
在高层应用中，用户通常使用现成的计算框架，它们可以通过 OpenACC 进行优化以加速大规模计算任务。

**主要组件**：
- **数值模拟和科学计算框架**：如 LAMMPS、GROMACS、ANSYS 等，它们在模拟大规模物理现象（如分子动力学、流体力学）时可以通过 OpenACC 加速特定的计算模块。
- **深度学习框架**：尽管 OpenACC 本身不是主流的深度学习加速技术，但某些框架可以通过集成 OpenACC 指令优化特定的计算内核。
- **HPC 应用框架**：如 OpenFOAM 和 WRF，这些高性能计算应用框架可以通过 OpenACC 进行并行化，以提高在多核和异构环境中的执行效率。

#### 3.7.2 运行时环境层

实现了使用 OpenACC 和 CUDA Runtime API 的 C 程序，用于获取和打印 CUDA 设备的信息。

这段代码的主要功能和要点如下：

- CUDA 设备数量获取：通过 `acc_get_num_devices` 获取系统中可用的 NVIDIA CUDA 设备数量，并打印出来。
- 设备属性查询：循环遍历每个设备，使用 `cudaGetDeviceProperties` 获取设备名称、计算能力和全局内存大小。
- 错误处理：使用 `cudaCheckError` 宏简化了对 CUDA 函数调用的错误检查。
- CUDA 驱动版本获取：通过 `cudaDriverGetVersion` 获取当前 CUDA 驱动的版本信息并打印。

```c++
#include <stdio.h>
#include <openacc.h>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define cudaCheckError(call)                                                    \
    {                                                                           \
        cudaError_t cudaStatus = call;                                          \
        if (cudaStatus != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error: %s at line %d\n",                      \
                    cudaGetErrorString(cudaStatus), __LINE__);                  \
            exit(cudaStatus);                                                   \
        }                                                                       \
    }

int main() {
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    printf("Total CUDA devices found: %d\n", num_devices);

    for (int device_id = 0; device_id < num_devices; device_id++) {
        acc_set_device_num(device_id, acc_device_nvidia);

        // 使用 CUDA Runtime API 获取设备信息
        cudaDeviceProp deviceProp;
        cudaCheckError(cudaGetDeviceProperties(&deviceProp, device_id));

        printf("\nDevice %d: %s\n", device_id, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %.2f GB\n", (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    }

    // 获取 CUDA 驱动版本
    int driver_version = 0;
    cudaCheckError(cudaDriverGetVersion(&driver_version));
    printf("\nCUDA Driver version: %d\n", driver_version / 1000);

    return 0;
}

```

结果：

```
Total CUDA devices found: 1

Device 0: NVIDIA GeForce RTX 4080 SUPER
  Compute capability: 8.9
  Total global memory: 15.70 GB

CUDA Driver version: 12
```

#### 3.7.3 计算库层

OpenACC 提供了一组指令和库，使开发者能够方便地将现有代码加速。

OpenACC 编程模型是一种指令式的并行编程框架，旨在帮助开发人员将现有的串行代码迁移到并行环境中，从而实现更高的性能。该模型包含几个关键概念：

1. **数据并行与任务并行**：OpenACC 支持数据并行和任务并行两种方式。数据并行涉及将数据分割成多个部分，并在不同处理器上同时处理这些数据；而任务并行则是将不同的任务划分为多个部分，并在多个处理器上同时执行这些任务。

2. **编译器指令**：OpenACC 使用指令来指定代码块的并行执行方式。开发人员可以在现有代码中插入这些指令，以实现并行计算，指令通常由编译器生成，并以 `#pragma acc` 语法表示。

3. **主要 OpenACC 指令**：OpenACC 提供多种指令以支持不同类型的并行计算。其中一些主要指令包括：`parallel`：用于并行执行一个代码块。`kernels`：在多个处理器上并行执行多个任务。`loop`：在多个处理器上并行执行循环。`data`：指定数据在不同处理器上的存储方式。`enter data` 和 `exit data`：管理数据传输和内存分配。
   
4. **指令参数与子句**：OpenACC 指令通常包含参数和子句，以指定执行方式及其他相关信息。例如，`parallel` 指令可以使用 `num_gangs`、`num_workers` 和 `vector_length` 等参数来详细说明并行执行的方式。

5. **运行时函数与环境变量**：OpenACC 还提供一些运行时函数和环境变量，用于控制并行计算的执行方式及性能。例如，开发人员可以使用 `acc_set_device_num()` 函数来设置使用的处理器编号。

数据并行和任务并行是并行计算中的两种基本模式，它们的主要区别在于并行计算的基本单位。

##### 数据并行：
数据并行是一种将数据划分为多个部分，并在不同处理器上同时处理这些数据的模式。在这种模式中，每个处理器执行相同的操作，但处理的数据输入和输出各不相同。数据并行通过将数据分割成块或子集，使不同的处理器能够同时处理这些块或子集。示例：在矩阵乘法中，可以将矩阵划分为多个块，并将每个块分配给不同的处理器。各个处理器同时执行相同的乘法操作，最后将结果合并以得到最终的矩阵乘积。

##### 任务并行：
任务并行则是将不同的任务划分为多个部分，并在不同处理器上同时执行这些任务的模式。在这种模式中，每个处理器执行不同的操作，但所用的输入和输出数据相同。任务并行通过将不同的任务分配给不同的处理器来实现。示例：在图像处理领域，可以将多种图像处理操作（如滤波、边缘检测等）划分为多个任务，并将这些任务分配给不同的处理器。各个处理器同时执行各自的操作，最终将结果合并得到处理后的图像。

OpenACC指令可以插入到C/C++或Fortran代码中的任何位置。通常情况下，OpenACC指令应该紧接着放在代码块的前面，例如：

```c++
arduinoCopy code#pragma acc parallel loop
for (int i = 0; i < N; i++)
{
    // parallel code block
}
```

在上面的示例中，使用#pragma acc parallel loop指令来指定代码块的并行执行方式，并在for循环之前插入这个指令。

除了可以在代码块前面插入OpenACC指令外，还可以在函数前面或文件开头使用OpenACC指令来指定整个文件或函数的默认并行执行方式，例如：

```c++
csharpCopy code#pragma acc data copyin(A[0:N], B[0:N]) copyout(C[0:N])
void my_function()
{
    #pragma acc parallel loop
    for (int i = 0; i < N; i++)
    {
        // parallel code block
    }
}
```

在上述示例中，使用 `#pragma acc data` 指令来设定默认的数据传输方式，而使用 `#pragma acc parallel loop` 指令来指明 `for` 循环的并行执行方式。这些指令可以插入在函数前或文件开头，以定义整个文件或函数的默认并行执行策略。

循环嵌套是指在一个循环结构内部包含另一个循环结构，从而形成多层嵌套的循环。这种结构在编程中非常常见，尤其用于处理多维数组和矩阵等数据结构。在并行计算中，循环嵌套同样是一个常见的结构，可以通过循环指令将嵌套循环转换为并行计算，从而提升程序的性能。嵌套循环的层数越多，程序的计算复杂度就越高。在进行并行计算时，需将嵌套循环转换为并行计算结构，以便将计算任务分配给多个线程并行处理。通常，这涉及使用多个循环指令，以有效地将计算任务分配到不同的线程上。

下面是一个简单的嵌套循环结构，用于计算矩阵乘法：

```c++
cssCopy codefor (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

该循环结构包含三层嵌套循环，用于计算矩阵乘法。在进行并行计算时，可以使用 `collapse` 指令将这三层嵌套循环合并为一个单层循环，然后利用 `gang`、`worker` 和 `vector` 等指令将其转化为并行计算结构。例如，可以使用以下指令将上述循环结构转换为并行计算结构：

```
cssCopy code#pragma acc data copyin(A[0:N][0:N], B[0:N][0:N]) copyout(C[0:N][0:N])
#pragma acc kernels collapse(3) gang worker vector
{
    #pragma acc loop gang
    for (int i = 0; i < N; i++) {
        #pragma acc loop worker
        for (int j = 0; j < N; j++) {
            float temp = 0;
            #pragma acc loop vector reduction(+:temp)
            for (int k = 0; k < N; k++) {
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}
```

在上述代码中，使用 `data` 指令结合 `copyin` 和 `copyout` 子句将矩阵 A、B 和 C 从主机内存复制到加速器内存。同时，使用 `kernels` 指令和 `collapse` 子句将三层嵌套循环转换为单层循环。接着，使用 `gang`、`worker` 和 `vector` 等指令将循环转变为并行计算结构，从而有效提升计算性能。

#### 3.7.4 编程模型和语言层

下面这段代码主要演示了如何使用 OpenACC 在并行环境中安全地对数组进行多种操作，实现了对一个数组进行并行处理，主要功能包括读取、写入、捕获和更新操作。并通过原子操作防止数据竞争。

- 创建并初始化一个大小为 100 的数组 `data`，内容为 0 到 99。

- 在并行环境中，检查数组元素值是否大于等于 50，若是，则将该元素的索引值赋给 `readSum`。

- 在并行环境中，检查数组元素值是否大于等于 50，若是，则计算 `x * 2 + 1` 并将其赋值给 `writeSum`。

- 在并行环境中，检查数组元素值是否大于等于 50，若是，则将该元素的值赋给 `captureSum`，并将该元素自减 1（减少其值）。

- 在并行环境中，检查数组元素值是否大于等于 50，若是，则对 `updateSum` 进行自增操作，计算符合条件的元素个数。

- 最后输出 `captureSum` 的值，这个值是从数组中捕获的元素值，并在捕获后减少了对应的元素。

```c++
#include "iostream"
#include "stdlib.h"

int main(){
    int n = 100;
    double * data = (double *)malloc( n * sizeof(double));
    for( int x = 0; x < n; ++x){
	    data[x] = x;	    
    }

    double readSum = 0.0;
    double writeSum = 0.0;
    double captureSum = 0.0;
    double updateSum = 0.0;

    // the atomic construct prevents reading the value while a gang/worker/vector is writing and vice versa
    // this is the read clause read the value of one variable into another variable
    #pragma acc parallel loop copy(data[0:n]) copyout(readSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
            #pragma acc atomic read
            readSum = x;
        }
    }

    // the atomic construct prevents reading the value while a gang/worker/vector is writing and vice versa
    // this is the write clause that only allows a direct write to a variable from a expression
    #pragma acc parallel loop copy(data[0:n]) copyout(writeSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
            #pragma acc atomic write
            writeSum = x*2 + 1;
        }
    }

    //this is the capture clause that the update to the expression into another variable
    #pragma acc parallel loop copy(data[0:n]) copyout(captureSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
            #pragma acc atomic capture
            captureSum = data[x]--;
            //std::cout << captureSum << ". " << data[x] << ". " << x << std::endl;
            }
    }

    std::cout << captureSum << std::endl;

    //this is the update clause which locks the update of a particualar variable from certain operations
    #pragma acc parallel loop copy(data[0:n]) copyout(updateSum)
    for(int x = 0; x < n; ++x){
        if(data[x] >= n/2){
            #pragma acc atomic update
    	    updateSum++;
        }
    }
    return 0;
}

```

结果：

```
99
```

下面这段代码实现了一个二维卷积操作，主要用于处理图像数据。通过使用 OpenACC 的 `#pragma acc parallel loop` 指令，代码实现了对二维卷积操作的并行化处理。这使得程序能够充分利用现代多核处理器或 GPU 的计算能力，从而加速卷积计算。

- **pragma acc parallel loop**：指示编译器将接下来的循环并行化执行。这个指令使得 `for` 循环在多个线程中并行运行，利用多核 CPU 或 GPU 进行加速。
- **collapse(2)**：这个选项指示编译器将嵌套的两个循环（外层和内层循环）进行合并，以形成一个更大的循环。这有助于提高并行化的效率，因为它允许编译器更好地分配迭代工作负载。
- **present(input, kernel, output)**：这个选项告知编译器 `input`、`kernel` 和 `output` 数据已经存在于设备（如 GPU）内存中，避免了在计算前进行数据拷贝，从而减少了内存传输的开销。
- 使用一个二维数组作为输入数据（`input`），并定义一个卷积核（`kernel`）。卷积操作通过将卷积核在输入数据上滑动，计算局部区域的加权和，生成输出数据（`output`）。使用 OpenACC 的 `#pragma acc parallel loop` 指令进行并行处理。通过嵌套循环，将卷积核与输入数据相乘，并累加到 `sum` 中。最后将计算结果赋值给输出矩阵。
- 在这段代码中，卷积操作的核心是对每个输出元素的计算都是独立的，意味着不同的线程可以同时计算不同的输出元素。因此，OpenACC 非常适合用于这种类型的计算密集型任务。

```c++
#include <iostream>
#include <vector>

#define WIDTH 5
#define HEIGHT 5
#define KERNEL_SIZE 3

void convolution2D(const std::vector<std::vector<float>>& input,
                   const std::vector<std::vector<float>>& kernel,
                   std::vector<std::vector<float>>& output) {
    int inputWidth = input[0].size();
    int inputHeight = input.size();
    int kernelSize = kernel.size();
    
    // Initialize output matrix with zeros
    for (int i = 0; i < inputHeight - kernelSize + 1; ++i) {
        for (int j = 0; j < inputWidth - kernelSize + 1; ++j) {
            output[i][j] = 0;
        }
    }

    // Perform convolution
    #pragma acc parallel loop collapse(2) present(input, kernel, output)
    for (int i = 0; i < inputHeight - kernelSize + 1; ++i) {
        for (int j = 0; j < inputWidth - kernelSize + 1; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < kernelSize; ++ki) {
                for (int kj = 0; kj < kernelSize; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
}

int main() {
    // Example input and kernel
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 0, 1},
        {4, 5, 6, 1, 0},
        {7, 8, 9, 0, 1},
        {0, 1, 2, 1, 0},
        {1, 0, 1, 2, 3}
    };

    std::vector<std::vector<float>> kernel = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };

    std::vector<std::vector<float>> output(HEIGHT - KERNEL_SIZE + 1, std::vector<float>(WIDTH - KERNEL_SIZE + 1, 0));

    convolution2D(input, kernel, output);

    // Print output matrix
    for (const auto& row : output) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

```

结果：

```
1.0000 0.0100 0.0200 0.0300 0.0400 0.0500 0.0600 0.0700 0.0800 0.0900 
0.0100 0.9999 0.0300 0.0400 0.0500 0.0600 0.0700 0.0800 0.0900 0.1000 
0.0200 0.0298 0.9994 0.0500 0.0600 0.0700 0.0800 0.0900 0.1000 0.1100 
0.0300 0.0397 0.0482 0.9976 0.0700 0.0800 0.0900 0.1000 0.1100 0.1200 
0.0400 0.0496 0.0578 0.0642 0.9942 0.0900 0.1000 0.1100 0.1200 0.1300 
0.0500 0.0595 0.0673 0.0731 0.0769 0.9890 0.1100 0.1200 0.1300 0.1400 
0.0600 0.0694 0.0768 0.0819 0.0850 0.0861 0.9820 0.1300 0.1400 0.1500 
0.0700 0.0793 0.0863 0.0908 0.0930 0.0932 0.0920 0.9733 0.1500 0.1600 
0.0800 0.0892 0.0958 0.0997 0.1010 0.1003 0.0980 0.0948 0.9632 0.1700 
0.0900 0.0991 0.1053 0.1085 0.1091 0.1074 0.1041 0.0998 0.0951 0.9518 
```



## 4. AMD 平台

[按照 NVIDIA 平台的结构填写 AMD 平台的内容]

## 5. Intel 平台

[按照 NVIDIA 平台的结构填写 Intel 平台的内容]

## 6. 算能 TPU 平台

### 6.1 SOPHON SDK

#### 6.1.1 技术栈架构

1. **系统软件层**
   - SOPHON设备驱动：为TPU提供基本的系统级支持（类似于NVIDIA GPU驱动）
   - TPU-Kernel：基于SOPHON BM1684、BM1684X底层原子操作接口的底层编程接口（类似于CUDA Driver API）
     - 需要用户熟悉设备硬件架构和指令集
     - 提供与SOPHON TPU硬件交互的底层接口
     - 适用于需要细粒度控制的高级应用

2. **运行时环境层**
   - BMLib：提供基础接口，包括设备Handle管理、内存管理、数据搬运、API发送和同步等（类似于CUDA Runtime API的部分功能）
   - BMRuntime：用于模型推理的运行时库（提供了类似CUDA Runtime API的高级抽象）
     - 简化了TPU的使用
     - 自动处理许多底层细节

3. **编程模型和语言层**
   - BMLang：基于C++的面向SOPHON智能视觉深度学习处理器的高级编程库（类似于CUDA C/C++的角色）
     - 使用张量数据(bmlang::Tensor)和计算操作(bmlang::Operator)编写代码
     - 通过bmlang::compile或bmlang::compile_with_check生成可运行的BModel
     - 支持在TPU和CPU上混合编程
   - TPU-MLIR：支持将PyTorch、ONNX等框架模型转换为TOP MLIR，然后lowering到TPU MLIR，最后部署到BModel
   - TPU-NNTC：支持多种框架模型的转换和量化，生成可在TPU上运行的BModel
4. **计算库层**
   - BMCV：提供张量运算及图像处理功能，如色彩空间转换、尺度变换、仿射变换等（类似于cuBLAS和其他CUDA专用算子库）
   - SOPHON-MW：支持SOPHON设备硬件加速的多媒体库，包括SOPHON-OpenCV和SOPHON-FFmpeg

5. **框架模型层**
   - SAIL (Sophon Artificial Intelligence Library)：支持Python/C++的高级接口（类似于PyTorch和TensorFlow对CUDA的支持）
     - 对BMRuntime、BMCV、sophon-mw等底层库接口的封装
     - 简化了TPU编程，提供更高级的抽象
   - PyTorch、TensorFlow等框架：通过TPU-MLIR或TPU-NNTC工具链支持这些框架模型的转换和优化，以在TPU上运行


总的来说，算能TPU的架构在很多方面与CUDA相似，都提供了从底层硬件接口到高级框架支持的完整堆栈。主要区别在于TPU更专注于深度学习处理器的优化，并提供了专门的模型编译和优化工具。

![alt text](ONNX.webp)

图片展示了算能TPU的软件栈架构，从顶层的深度学习框架到底层的硬件加速器，在图片中，SAIL位于中间层，作为运行时环境的一部分。而在AI技术栈中，SAIL被放在较高的"框架模型层"。这种差异反映了SAIL的多功能性：
   - 作为运行时环境：SAIL提供了对底层硬件和库的封装，使其能够高效地执行编译后的模型。
   - 作为高级接口：SAIL也提供了Python/C++的高级API，使开发者能够更容易地使用TPU的功能。

这种双重角色解释了为什么SAIL可以在不同的架构描述中出现在不同的层次。在实际应用中，SAIL既是连接高层框架和底层硬件的桥梁，也是开发者直接使用的高级接口。

#### 6.1.2 系统软件层

自定义算子的过程


#### 6.1.3 运行时环境层

#### 6.1.4 编程模型和语言层

##### TPU-MLIR 介绍

TPU-MLIR 是算能深度学习处理器的编译器工程，提供了一套完整的工具链，用于将不同框架下预训练的神经网络转化为可以在算能智能视觉深度学习处理器上高效运行的模型文件（BModel/CviModel）。

##### 主要特点

1. 支持多种框架：直接支持 PyTorch、ONNX、TFLite 和 Caffe 等框架模型的转换。
2. 开源：代码已开源到 GitHub（https://github.com/sophgo/tpu-mlir）。
3. 学术支持：有相关论文描述其整体设计思路（https://arxiv.org/abs/2210.15016）。

##### 架构概览

TPU-MLIR 的整体架构包括以下主要组件：

1. 前端转换：将各种框架的模型转换为 MLIR 表示。
2. 优化阶段：对 MLIR 进行各种优化。
3. 后端生成：生成可在 TPU 上运行的 BModel/CviModel。

#####  使用流程

###### 1. 模型转换

使用 `model_transform` 工具将原始模型转换成 MLIR 文件。

###### 2. 量化（可选）

如需 INT8 量化：
- 使用 `run_calibration` 生成校准表。
- 使用 `run_qtable` 生成量化表（用于决定哪些层采用浮点计算）。

###### 3. 模型部署

使用 `model_deploy` 将 MLIR 文件转换成 BModel/CviModel。

##### Lowering 过程
TPU-MLIR使用两种主要的方言：TOP（Tensor Operator）和TPU。

- TOP方言：
  - 硬件无关层
  - 支持F32/F16/BF16/INT8对称/INT8非对称等类型
  - 代表了网络的高级表示

- TPU方言：
  - 硬件相关层
  - 针对特定TPU硬件优化
  - 包含了硬件特定的量化和优化策略



Lowering是将TOP层OP下沉到TPU层OP的过程：
- 将算子从硬件无关层(TOP)转换到硬件相关层(TPU)
- 支持F32/F16/BF16/INT8对称/INT8非对称等类型转换
- 涉及量化算法，针对不同硬件有不同实现
- 处理混合精度情况，在需要时插入CastOp
Lowering 是将 TOP 层 OP 下沉到 TPU 层 OP 的过程：

- 支持 F32/F16/BF16/INT8 对称/INT8 非对称等类型转换
- 处理混合精度情况，在需要时插入 CastOp

##### CodeGen 过程

CodeGen 是将 MLIR 文件转换为最终 BModel 的过程，主要包括：

1. 指令生成：执行不同 op 的 CodeGen 接口，生成相应的二进制指令
2. 指令存储：使用 store_cmd 将指令存储在指定数据结构中
3. 指令取出：所有 op 的二进制码生成完毕后，调用 BM168X 系列类中封装的函数取出指令，最终生成 BModel

##### 后端实现

- 使用动态库（libbackend_xxx.so）封装硬件后端
- 通过函数指针加载后端函数
- 使用 EngineStorer 和 CmdStorer 系列类管理指令存储
- 采用单例模式和装饰器模式实现灵活的指令管理

通过这种设计，TPU-MLIR 能够有效地将各种深度学习框架的模型转换为可在 TPU 上高效运行的 BModel，同时提供了灵活的优化和定制空间。




基于提供的上下文，我可以为您综合一个关于TPU-MLIR的介绍，包括TOP和TPU两种方言，以及CodeGen的过程：

##### 自定义算子开发

TPU-MLIR 支持添加自定义算子，主要步骤如下：

###### 1. 前端定义

使用 TpuLang 接口定义自定义算子：

```python
import transform.TpuLang as tpul

tpul.init("BM1684X", True)

# 定义输入
x = tpul.Tensor(dtype="float32", shape=[1, 3, 224, 224], name="input")

# 添加自定义算子
def shape_func(tensors_in):
    return [tensors_in[0].shape]

outs = tpul.custom(
    tensors_in=[x],
    shape_func=shape_func,
    op_name="custom_op_name",
    params={"param1": value1, "param2": value2},
    out_dtypes=["float32"],
    out_names=["custom_out"]
)

# 编译生成 MLIR
tpul.compile("model_name", [x], outs, False, 2, has_custom=True)
```

###### 2. 后端实现

1. 在 `$TPUC_ROOT/customlayer/include` 添加头文件。
2. 在 `$TPUC_ROOT/customlayer/src` 添加实现文件。
3. 在 `backend_custom_param.h` 定义参数结构体。
4. 添加 API 接口文件。
5. 在 `backend_custom_api.cpp` 定义后端调用接口。
6. 运行 `$TPUC_ROOT/customlayer/build.sh` 编译生成动态库。


TPU-MLIR 提供了一个强大的工具链，支持从多种深度学习框架到 TPU 可执行模型的转换。通过支持自定义算子，它为开发者提供了极大的灵活性，使得复杂的深度学习模型能够在 TPU 上高效运行。结合 TPUPerf 工具，开发者可以全面优化和验证其模型性能。

#### 6.1.5 计算库层

#### 6.1.6 框架模型层

#### 6.1.7 对比与思考

同样都是mlir，对于triton的思考





## 7. 摩尔线程平台

[按照 NVIDIA 平台的结构填写摩尔线程平台的内容]

## 8. 总结与展望

[此处填写总结与展望内容]
异构计算

通用算子表达

---
## 9. AI技术栈安装指南


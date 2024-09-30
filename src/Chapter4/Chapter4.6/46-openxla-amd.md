# OpenXLA (AMD)
![OpenXLA 生态系统](https://github.com/openxla/xla/raw/main/docs/images/openxla.svg)
### AMD运行
兼容性后端：OpenXLA支持英伟达的CUDA和AMD的ROCm后端，通过不同的硬件驱动程序优化计算性能。

MLIR编译器基础：使用MLIR编译框架，OpenXLA能将高层次的机器学习模型转换为适用于不同硬件平台的低级代码。

设备无关接口：OpenXLA提供统一的接口，屏蔽硬件差异，使同一模型能在不同的GPU架构上无缝切换。



OpenXLA 是一个为多种硬件设备加速深度学习和机器学习模型执行的开放框架。它的设计目的是在不同硬件平台（如GPU、TPU、CPU和加速卡）上优化机器学习工作负载。OpenXLA 由多个子组件组成，这些组件为不同层次的优化和执行提供支持。

# Triton (AMD)

- **AMD 平台**：Triton 虽然支持 AMD GPU，但由于缺少特定的硬件优化，推理性能可能不如在 NVIDIA GPU 上表现出色。AMD 平台上的 Triton 适合小规模或对性能要求较低的任务，但开发者可能需要手动进行更多的性能调优。

- **NVIDIA 平台**：作为 NVIDIA 开发的推理服务器，Triton 在 NVIDIA GPU 上具有显著的性能优势。深度集成的 CUDA 和 TensorRT 支持，使得 Triton 在大规模高效推理任务中占据主导地位。NVIDIA 提供的优化工具链也让开发者能快速优化和部署推理模型。

总体来说，Triton 在 AMD 平台上支持基本的推理功能，但性能上更倾向于NVIDIA平台。如果开发者需要在大规模或高性能推理场景下工作，NVIDIA 平台由于其针对 Triton 的优化，仍然是更具优势的选择。
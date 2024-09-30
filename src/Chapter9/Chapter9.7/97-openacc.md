# OpenACC


```bash
#!/bin/bash

# 步骤 1: 下载并解压安装 NVIDIA HPC SDK
echo "下载并安装 NVIDIA HPC SDK..."
wget https://developer.download.nvidia.com/hpc-sdk/24.7/nvhpc_2024_247_Linux_x86_64_cuda_multi.tar.gz
tar xpzf nvhpc_2024_247_Linux_x86_64_cuda_multi.tar.gz
cd nvhpc_2024_247_Linux_x86_64_cuda_multi
sudo ./install

# 步骤 2: 设置环境变量
echo "设置环境变量..."
echo 'export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 步骤 3: 创建测试代码
echo "创建测试代码 matrix_add.c..."
cat <<EOL > matrix_add.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

#define N 10000

void matrix_add_cpu(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

void matrix_add_gpu(float *A, float *B, float *C, int n) {
    #pragma acc parallel loop collapse(2) copyin(A[0:n*n], B[0:n*n]) copyout(C[0:n*n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

int main() {
    float *A, *B, *C;
    A = (float*) malloc(N * N * sizeof(float));
    B = (float*) malloc(N * N * sizeof(float));
    C = (float*) malloc(N * N * sizeof(float));

    srand(time(0));
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start_cpu = clock();
    matrix_add_cpu(A, B, C, N);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\\n", cpu_time);

    clock_t start_gpu = clock();
    matrix_add_gpu(A, B, C, N);
    clock_t end_gpu = clock();
    double gpu_time = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    printf("GPU Time: %f seconds\\n", gpu_time);

    free(A);
    free(B);
    free(C);

    return 0;
}
EOL

# 步骤 4: 编译并运行测试程序
echo "编译并运行测试程序..."
pgcc -acc -Minfo=accel -o matrix_add matrix_add.c
./matrix_add
```

### 脚本说明：
1. **下载并安装NVIDIA HPC SDK**：该脚本会从NVIDIA官网下载最新的HPC SDK（24.7版本），解压并进行安装。
2. **设置环境变量**：将HPC SDK的编译器路径和库路径添加到系统的`PATH`和`LD_LIBRARY_PATH`中，并立即生效。
3. **创建测试代码**：生成一个`matrix_add.c`的OpenACC测试代码，其中包含矩阵加法的CPU和GPU实现。
4. **编译并运行测试代码**：使用`pgcc`编译器编译该OpenACC代码，并输出运行时间来对比CPU和GPU的执行时间。

/*
 * 第17章 代码示例：C++20 consteval 在CUDA中的使用
 * 编译：nvcc -std=c++20 -arch=sm_70 cpp20_consteval_example.cu -o cpp20_consteval
 *
 * 演示consteval函数如何实现跨执行空间调用
 */

#include <cuda_runtime.h>
#include <stdio.h>

// =============================================================
// 示例1：consteval host函数被device/global函数调用
// =============================================================

// 编译期在主机端求值的常量函数
consteval int hostConstantValue() {
    return 42;
}

consteval float hostPiConstant() {
    return 3.14159265359f;
}

__device__ float deviceFunctionUsingHostConstant() {
    // OK: consteval host函数可以在device函数中调用
    return hostPiConstant() * 2.0f;
}

__global__ void kernelUsingHostConstant(float *result) {
    // OK: consteval host函数可以在global函数中调用
    *result = hostConstantValue() + deviceFunctionUsingHostConstant();
}

// =============================================================
// 示例2：consteval device函数被host函数调用
// =============================================================

// 编译期在设备端求值的常量函数
consteval __device__ int deviceConstantValue() {
    return 100;
}

consteval __device__ float deviceComputeRatio() {
    return 3.0f / 4.0f;
}

// 主机函数可以调用consteval device函数
float hostFunctionUsingDeviceConstant() {
    // OK: consteval device函数可以在host函数中调用
    return deviceConstantValue() * deviceComputeRatio();
}

__global__ void kernelUsingDeviceConstant(float *result) {
    *result = deviceConstantValue() + deviceComputeRatio();
}

// =============================================================
// 示例3：C++20三路比较运算符
// =============================================================

struct Point3D {
    float x, y, z;

    // C++20: 默认三路比较
    auto operator<=>(const Point3D&) const = default;
};

__device__ int comparePoints(const Point3D& a, const Point3D& b) {
    if (a < b) return -1;
    if (b < a) return 1;
    return 0;
}

__global__ void testSpaceshipOperator(int *result) {
    Point3D p1 = {1.0f, 2.0f, 3.0f};
    Point3D p2 = {1.0f, 2.0f, 4.0f};
    *result = comparePoints(p1, p2);
}

// =============================================================
// 主函数
// =============================================================

int main() {
    cudaError_t err;
    int testPassed = 1;

    // 测试1：consteval host函数在kernel中的使用
    printf("Test 1: consteval host function in device code\n");
    {
        float *d_result;
        cudaMalloc(&d_result, sizeof(float));
        kernelUsingHostConstant<<<1, 1>>>(d_result);

        float h_result;
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        float expected = 42.0f + 3.14159265359f * 2.0f;
        printf("  Result: %f, Expected: %f\n", h_result, expected);

        if (fabsf(h_result - expected) < 1e-5f) {
            printf("  PASSED\n");
        } else {
            printf("  FAILED\n");
            testPassed = 0;
        }
        cudaFree(d_result);
    }

    // 测试2：consteval device函数在host代码中的使用
    printf("Test 2: consteval device function in host code\n");
    {
        float result = hostFunctionUsingDeviceConstant();
        float expected = 100.0f * 0.75f;
        printf("  Result: %f, Expected: %f\n", result, expected);

        if (fabsf(result - expected) < 1e-5f) {
            printf("  PASSED\n");
        } else {
            printf("  FAILED\n");
            testPassed = 0;
        }
    }

    // 测试3：三路比较运算符
    printf("Test 3: Three-way comparison operator in device code\n");
    {
        int *d_result;
        cudaMalloc(&d_result, sizeof(int));
        testSpaceshipOperator<<<1, 1>>>(d_result);

        int h_result;
        cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        printf("  Result: %d, Expected: -1\n", h_result);

        if (h_result == -1) {
            printf("  PASSED\n");
        } else {
            printf("  FAILED\n");
            testPassed = 0;
        }
        cudaFree(d_result);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        testPassed = 0;
    }

    printf("\n=== Overall: %s ===\n",
           testPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return testPassed ? 0 : 1;
}

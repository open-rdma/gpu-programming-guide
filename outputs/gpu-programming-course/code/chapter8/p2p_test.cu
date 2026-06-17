#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

int main() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        printf("This test requires at least 2 CUDA devices. Found %d.\n", deviceCount);
        return 0;
    }

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    // 打印设备信息
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s (Compute %d.%d, %.2f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // 检查设备 0 和 1 之间的 P2P 能力
    int canAccess01, canAccess10;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess10, 1, 0));

    printf("\nP2P Access: Device 0 -> Device 1: %s\n",
           canAccess01 ? "Supported" : "Not Supported");
    printf("P2P Access: Device 1 -> Device 0: %s\n",
           canAccess10 ? "Supported" : "Not Supported");

    // 测试数据大小：16MB
    const size_t dataSize = 16 * 1024 * 1024;
    const size_t numFloats = dataSize / sizeof(float);
    float *d_data0, *d_data1;
    float *h_data;

    // 分配设备内存
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_data0, dataSize));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d_data1, dataSize));

    // 分配页锁定主机内存
    CUDA_CHECK(cudaMallocHost(&h_data, dataSize));

    // 初始化数据
    for (size_t i = 0; i < numFloats; i++) {
        h_data[i] = (float)i;
    }

    // 将数据拷贝到设备 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_data0, h_data, dataSize, cudaMemcpyHostToDevice));

    // ---------- 预创建设备 0 和设备 1 各自的事件对 ----------
    cudaEvent_t start0, stop0;   // 用于设备 0 上的计时
    cudaEvent_t start1, stop1;   // 用于设备 1 上的计时

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventCreate(&start0));
    CUDA_CHECK(cudaEventCreate(&stop0));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));

    float elapsedTime;

    // === 测试1: P2P 直接拷贝（使用 cudaMemcpyPeer）===
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(start0, 0));
    CUDA_CHECK(cudaMemcpyPeer(d_data1, 1, d_data0, 0, dataSize));
    CUDA_CHECK(cudaEventRecord(stop0, 0));
    CUDA_CHECK(cudaEventSynchronize(stop0));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start0, stop0));
    printf("\n=== P2P Memory Copy Performance ===\n");
    printf("P2P direct copy (cudaMemcpyPeer): %.3f ms, Bandwidth: %.2f GB/s\n",
           elapsedTime, (dataSize / (elapsedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

    // === 测试2: 通过主机中转的拷贝（分段计时并累加）===
    // 第一段：设备 0 -> 主机
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(start0, 0));
    CUDA_CHECK(cudaMemcpy(h_data, d_data0, dataSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop0, 0));
    CUDA_CHECK(cudaEventSynchronize(stop0));
    float timeD2H;
    CUDA_CHECK(cudaEventElapsedTime(&timeD2H, start0, stop0));

    // 第二段：主机 -> 设备 1
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaEventRecord(start1, 0));
    CUDA_CHECK(cudaMemcpy(d_data1, h_data, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop1, 0));
    CUDA_CHECK(cudaEventSynchronize(stop1));
    float timeH2D;
    CUDA_CHECK(cudaEventElapsedTime(&timeH2D, start1, stop1));

    float totalStagedTime = timeD2H + timeH2D;
    printf("Host-staged copy (D2H + H2D):   %.3f ms, Bandwidth: %.2f GB/s\n",
           totalStagedTime,
           (dataSize / (totalStagedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

    // === 测试3: 尝试启用 P2P 访问后的 UVA 直接访问 ===
    if (canAccess01 && canAccess10) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

        // 使用 cudaMemcpyDefault 进行 P2P 拷贝
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventRecord(start0, 0));
        CUDA_CHECK(cudaMemcpy(d_data1, d_data0, dataSize, cudaMemcpyDefault));
        CUDA_CHECK(cudaEventRecord(stop0, 0));
        CUDA_CHECK(cudaEventSynchronize(stop0));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start0, stop0));
        printf("P2P copy via cudaMemcpyDefault:   %.3f ms, Bandwidth: %.2f GB/s\n",
               elapsedTime,
               (dataSize / (elapsedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

        printf("\nP2P access enabled successfully. Same pointer can be used on both devices.\n");
    }

    // 清理资源
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d_data0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaEventDestroy(start0));
    CUDA_CHECK(cudaEventDestroy(stop0));
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));

    printf("\nTest completed successfully.\n");
    return 0;
}
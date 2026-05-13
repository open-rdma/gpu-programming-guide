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

    // Print device info
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("Device %d: %s (Compute %d.%d, %.2f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // Check P2P capability between device 0 and 1
    int canAccess01, canAccess10;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess10, 1, 0));

    printf("\nP2P Access: Device 0 -> Device 1: %s\n",
           canAccess01 ? "Supported" : "Not Supported");
    printf("P2P Access: Device 1 -> Device 0: %s\n",
           canAccess10 ? "Supported" : "Not Supported");

    // Test data size: 16MB
    const size_t dataSize = 16 * 1024 * 1024;  // 16 MB
    const size_t numFloats = dataSize / sizeof(float);
    float *d_data0, *d_data1;
    float *h_data;

    // Allocate device memory
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&d_data0, dataSize));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d_data1, dataSize));

    // Allocate page-locked host memory
    CUDA_CHECK(cudaMallocHost(&h_data, dataSize));

    // Initialize data
    for (size_t i = 0; i < numFloats; i++) {
        h_data[i] = (float)i;
    }

    // Copy data to device 0
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMemcpy(d_data0, h_data, dataSize, cudaMemcpyHostToDevice));

    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsedTime;

    // === Test 1: P2P direct copy using cudaMemcpyPeer ===
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUDA_CHECK(cudaMemcpyPeer(d_data1, 1, d_data0, 0, dataSize));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\n=== P2P Memory Copy Performance ===\n");
    printf("P2P direct copy (cudaMemcpyPeer): %.3f ms, Bandwidth: %.2f GB/s\n",
           elapsedTime, (dataSize / (elapsedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

    // === Test 2: Copy via host staging ===
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaEventRecord(start, 0));
    // Device 0 -> Host
    CUDA_CHECK(cudaMemcpy(h_data, d_data0, dataSize, cudaMemcpyDeviceToHost));
    // Host -> Device 1
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMemcpy(d_data1, h_data, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Host-staged copy (D2H + H2D):   %.3f ms, Bandwidth: %.2f GB/s\n",
           elapsedTime, (dataSize / (elapsedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

    // === Test 3: Try P2P access with UVA ===
    if (canAccess01 && canAccess10) {
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

        // Use cudaMemcpyDefault for P2P copy
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(d_data1, d_data0, dataSize, cudaMemcpyDefault));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        printf("P2P copy via cudaMemcpyDefault:   %.3f ms, Bandwidth: %.2f GB/s\n",
               elapsedTime,
               (dataSize / (elapsedTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0));

        printf("\nP2P access enabled successfully. Same pointer can be used on both devices.\n");
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(d_data0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nTest completed successfully.\n");
    return 0;
}

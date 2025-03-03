#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

// Kernel CUDA: Realiza la misma operación que en CPU
__global__ void vectorComputationGPU(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx] + sinf(A[idx]) + cosf(B[idx]);
    }
}

int main() {
    int N = 1000000000; // 1,000 millones de elementos para notar la diferencia
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    // Asignar memoria en el host (CPU)
    A = new int[N];
    B = new int[N];
    C = new int[N];

    // Inicializar los vectores en CPU
    for (int i = 0; i < N; i++) {
        A[i] = i % 1000;
        B[i] = (i * 2) % 1000;
    }

    // Asignar memoria en el device (GPU)
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copiar datos del host a la GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configurar el lanzamiento del kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Medir tiempo de ejecución en GPU
    auto start = std::chrono::high_resolution_clock::now();
    vectorComputationGPU<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Asegurar que la GPU termine antes de medir el tiempo final
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Copiar resultados de vuelta a la CPU
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar algunos resultados
    std::cout << "Resultado en GPU: " << C[0] << ", " << C[1] << std::endl;
    std::cout << "Tiempo de ejecución en GPU: " << duration.count() << " segundos" << std::endl;

    // Liberar memoria en GPU y CPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

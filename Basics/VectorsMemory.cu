#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorSumGlobal(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
        printf("Thread %d computed C[%d] = %d + %d = %d\n", idx, idx, A[idx], B[idx], C[idx]);
    }
}

__global__ void vectorSumShared(int *A, int *B, int *C, int N) {
    extern __shared__ int sharedMem[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    if (idx < N) {
        sharedMem[tx] = A[idx] + B[idx];
    }
    
    __syncthreads();

    if (idx < N) {
        C[idx] = sharedMem[tx];
        printf("Shared Thread %d computed C[%d] = %d\n", idx, idx, C[idx]);
    }
}

int main() {
    int N = 10;  // Reducido para depuración
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    
    // Obtener información de la GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Ejecutando en GPU: " << prop.name << " (Device " << device << ")\n";

    // Reservar memoria en CPU
    A = new int[N];
    B = new int[N];
    C = new int[N];

    // Inicializar vectores
    std::cout << "Vectores de entrada:\n";
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
        std::cout << "A[" << i << "] = " << A[i] << ", B[" << i << "] = " << B[i] << "\n";
    }

    // Reservar memoria en GPU
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copiar datos a la GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzar kernel con Memoria Global
    std::cout << "Ejecutando kernel con Memoria Global...\n";
    vectorSumGlobal<<<1, N>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Esperar a que termine el kernel

    // Copiar resultados a CPU
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Resultados con Memoria Global:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "C[" << i << "] = " << C[i] << "\n";
    }

    // Lanzar kernel con Memoria Compartida
    std::cout << "Ejecutando kernel con Memoria Compartida...\n";
    vectorSumShared<<<1, N, N * sizeof(int)>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copiar resultados a CPU
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Resultados con Memoria Compartida:\n";
    for (int i = 0; i < N; i++) {
        std::cout << "C[" << i << "] = " << C[i] << "\n";
    }

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

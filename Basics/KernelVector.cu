#include <iostream>
#include <cuda_runtime.h>

// Kernel que usa memoria global para sumar dos vectores
__global__ void vectorSumGlobal(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel que usa memoria compartida para optimizar la suma de vectores
__global__ void vectorSumShared(int *A, int *B, int *C, int N) {
    extern __shared__ int sharedMem[];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tx = threadIdx.x;

    if (idx < N) {
        sharedMem[tx] = A[idx] + B[idx];
    }
    
    __syncthreads(); // Sincronizar los hilos del bloque
    
    if (idx < N) {
        C[idx] = sharedMem[tx];
    }
}

int main() {
    int N = 100000000;
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    // Asignar memoria en el host
    A = new int[N];
    B = new int[N];
    C = new int[N];

    // Inicializar los vectores en el host
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Asignar memoria en el device (GPU)
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copiar los datos del host a la GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Lanzar el kernel que usa memoria global
    vectorSumGlobal<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar un par de resultados
    std::cout << "Suma Global: " << C[0] << ", " << C[1] << std::endl;

    // Lanzar el kernel que usa memoria compartida
    vectorSumShared<<<(N + 255) / 256, 256, 256 * sizeof(int)>>>(d_A, d_B, d_C, N);
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar un par de resultados
    std::cout << "Suma con Memoria Compartida: " << C[0] << ", " << C[1] << std::endl;

    // Liberar la memoria de la GPU y la CPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

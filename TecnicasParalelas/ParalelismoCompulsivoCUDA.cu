#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorSum(int *A, int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
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

    // Inicializar los vectores
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Asignar memoria en el dispositivo (GPU)
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copiar datos de la memoria del host a la memoria de la GPU
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configurar bloques e hilos
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Llamar al kernel en la GPU
    vectorSum<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Copiar el resultado de vuelta al host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar algunos resultados
    std::cout << "Resultado de la suma: " << C[0] << ", " << C[1] << std::endl;

    // Liberar la memoria
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

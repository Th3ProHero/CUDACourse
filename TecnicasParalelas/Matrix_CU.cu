#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N 4096 // Matrix (4096x4096)

__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        float value = 0;
        for (int k = 0; k < size; ++k) {
            value += A[row * size + k] * B[k * size + col];
            // Add Operators
            value = sin(value) + cos(value); // Trigonometric
        }
        C[row * size + col] = value;
    }
}

void initialize_matrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void matrixMultiplyCPU(float *A, float *B, float *C, int size) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            float value = 0;
            for (int k = 0; k < size; ++k) {
                value += A[row * size + k] * B[k * size + col];
                // Añadir operaciones adicionales para hacerlo más pesado
                value = sin(value) + cos(value); // Operaciones trigonométricas
            }
            C[row * size + col] = value;
        }
    }
}

int main() {
    float *h_A, *h_B, *h_C;  // Matrices en el host
    float *d_A, *d_B, *d_C;  // Matrices en el device
    cudaEvent_t start, stop;

    // Asignar memoria para las matrices
    h_A = (float*)malloc(N * N * sizeof(float));
    h_B = (float*)malloc(N * N * sizeof(float));
    h_C = (float*)malloc(N * N * sizeof(float));

    // Inicializar las matrices A y B con valores aleatorios
    initialize_matrix(h_A, N);
    initialize_matrix(h_B, N);

    // Asignar memoria en el device
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));

    // Copiar las matrices de la memoria del host a la memoria del device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Crear eventos para medir el tiempo en GPU
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Configuración de bloques y hilos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 16, N / 16);

    // Llamada al kernel para multiplicar las matrices en la GPU
    for (int i = 0; i < 10; ++i) { // Ejecutar el kernel 10 veces
        matrixMultiplyGPU<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    }

    // Sincronizar y registrar el tiempo en GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTimeGPU;
    cudaEventElapsedTime(&elapsedTimeGPU, start, stop);
    
    std::cout << "Tiempo en GPU: " << elapsedTimeGPU / 1000 << " segundos" << std::endl;

    // Copiar el resultado de vuelta al host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Medir el tiempo en CPU
    clock_t startCPU = clock();
    for (int i = 0; i < 10; ++i) { // Ejecutar la multiplicación en CPU 10 veces
        matrixMultiplyCPU(h_A, h_B, h_C, N);
    }
    clock_t endCPU = clock();
    double elapsedTimeCPU = double(endCPU - startCPU) / CLOCKS_PER_SEC;
    std::cout << "Tiempo en CPU: " << elapsedTimeCPU << " segundos" << std::endl;

    // Liberar la memoria del device y el host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
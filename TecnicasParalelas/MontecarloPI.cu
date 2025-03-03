#include <iostream>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define N 1410065408  // Número de puntos

// Kernel CUDA para generar puntos aleatorios y calcular la aproximación de PI
__global__ void monteCarloKernel(int *count, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Inicializar el generador de números aleatorios
        curandState state;
        curand_init(1234, id, 0, &state);
        
        // Generar números aleatorios entre 0 y 1 para las coordenadas (x, y)
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        // Comprobar si el punto cae dentro del círculo unitario
        if (x * x + y * y <= 1.0f) {
            atomicAdd(count, 1);  // Contar puntos dentro del círculo
        }
    }
}

int main() {
    int *d_count, h_count = 0;
    
    // Obtener información de la GPU
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    
    // Mostrar el nombre de la GPU que se está utilizando
    std::cout << "Ejecutando en la GPU: " << deviceProp.name << std::endl;

    // Asignar memoria en la GPU
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Configuración de los bloques e hilos
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Crear eventos para medir el tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Medir el tiempo de ejecución
    cudaEventRecord(start);
    
    // Lanzar el kernel
    std::cout << "Lanzando kernel de Monte Carlo..." << std::endl;
    monteCarloKernel<<<blocks, threadsPerBlock>>>(d_count, N);
    
    // Esperar a que el kernel termine
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcular el tiempo transcurrido
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copiar el resultado de vuelta al host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Calcular la aproximación de PI
    float pi = 4.0f * h_count / N;

    // Imprimir resultados
    std::cout << "Aproximación de PI: " << pi << std::endl;
    std::cout << "Puntos dentro del círculo: " << h_count << std::endl;
    std::cout << "Tiempo de ejecución en GPU: " << milliseconds << " ms" << std::endl;

    // Liberar memoria
    cudaFree(d_count);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

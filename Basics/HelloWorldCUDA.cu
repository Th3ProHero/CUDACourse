#include <iostream>
#include <cuda_runtime.h>

// Kernel que imprime un mensaje desde la GPU
__global__ void helloWorldFromGPU() {
    printf("¡Hola Mundo desde la GPU! Desde el hilo %d\n", threadIdx.x);
}

int main() {
    // Llamada al kernel con un solo bloque de 10 hilos
    helloWorldFromGPU<<<1, 10>>>();

    // Esperar a que los hilos terminen de ejecutar
    cudaDeviceSynchronize();

    return 0;
}

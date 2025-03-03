#include <iostream>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#define N 2048  // Tamaño de la cuadrícula
#define DT 0.1f  // Paso de tiempo
#define VISC 0.0001f  // Viscosidad del fluido
#define ITERACIONES 1000  // Más iteraciones para notar diferencia

// Kernel para difundir la velocidad del fluido
__global__ void diffuse(float *vx, float *vy, float *vx_new, float *vy_new) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * N + x;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        vx_new[index] = vx[index] + VISC * DT * (
            vx[index - 1] + vx[index + 1] + vx[index - N] + vx[index + N] - 4 * vx[index]);
        vy_new[index] = vy[index] + VISC * DT * (
            vy[index - 1] + vy[index + 1] + vy[index - N] + vy[index + N] - 4 * vy[index]);
    }
}

// Kernel para advectar la velocidad del fluido
__global__ void advect(float *vx, float *vy, float *vx_new, float *vy_new) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * N + x;

    if (x > 0 && x < N - 1 && y > 0 && y < N - 1) {
        float x_prev = x - DT * vx[index];
        float y_prev = y - DT * vy[index];

        int x0 = (int)x_prev, y0 = (int)y_prev;
        int x1 = x0 + 1, y1 = y0 + 1;

        vx_new[index] = 0.5f * (vx[y0 * N + x0] + vx[y1 * N + x1]);
        vy_new[index] = 0.5f * (vy[y0 * N + x0] + vy[y1 * N + x1]);
    }
}

int main() {
    float *vx, *vy, *vx_new, *vy_new;
    float *d_vx, *d_vy, *d_vx_new, *d_vy_new;

    size_t size = N * N * sizeof(float);
    vx = new float[N * N]();
    vy = new float[N * N]();
    vx_new = new float[N * N]();
    vy_new = new float[N * N]();

    cudaMalloc(&d_vx, size);
    cudaMalloc(&d_vy, size);
    cudaMalloc(&d_vx_new, size);
    cudaMalloc(&d_vy_new, size);

    cudaMemcpy(d_vx, vx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, vy, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // Configurar SFML para la visualización
    sf::RenderWindow window(sf::VideoMode(800, 800), "Simulación de Fluidos CUDA");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Loop de simulación y visualización
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        cudaEventRecord(start);
        for (int i = 0; i < ITERACIONES; i++) {
            diffuse<<<numBlocks, threadsPerBlock>>>(d_vx, d_vy, d_vx_new, d_vy_new);
            advect<<<numBlocks, threadsPerBlock>>>(d_vx_new, d_vy_new, d_vx, d_vy);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Actualizar la cuadrícula de velocidades en la GPU
        cudaMemcpy(vx, d_vx, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy, d_vy, size, cudaMemcpyDeviceToHost);

        window.clear();  // Limpiar la ventana

        // Dibujar los resultados
        for (int y = 0; y < N; y += N / 800) {
            for (int x = 0; x < N; x += N / 800) {
                int index = y * N + x;
                sf::RectangleShape pixel(sf::Vector2f(1, 1));
                pixel.setPosition(x * 800 / N, y * 800 / N);

                // Usar la velocidad para determinar el color del píxel
                float color = 255 * sqrt(vx[index] * vx[index] + vy[index] * vy[index]);
                pixel.setFillColor(sf::Color(color, 0, 255 - color));  // Rojo a azul

                window.draw(pixel);
            }
        }

        window.display();  // Mostrar los resultados en la ventana

        // Espera para que la simulación avance en la siguiente iteración
        cudaMemcpy(vx, d_vx_new, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(vy, d_vy_new, size, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vx_new);
    cudaFree(d_vy_new);
    delete[] vx;
    delete[] vy;
    delete[] vx_new;
    delete[] vy_new;

    return 0;
}

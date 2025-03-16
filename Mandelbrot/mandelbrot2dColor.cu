#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>

#define WIDTH 900
#define HEIGHT 900
#define MAX_ITER 1000  // Aumentando el número de iteraciones

// Función para verificar si un número pertenece al conjunto de Mandelbrot
__device__ int mandelbrot(double real, double imag) {
    double zr = real;
    double zi = imag;
    int n;
    for (n = 0; n < MAX_ITER; n++) {
        if (zr * zr + zi * zi > 4.0) {
            break;
        }
        double temp = zr * zr - zi * zi + real;
        zi = 2.0 * zr * zi + imag;
        zr = temp;
    }
    return n;
}

// Kernel de CUDA para generar la imagen de Mandelbrot
__global__ void mandelbrot_kernel(int *image, double minReal, double maxReal, double minImag, double maxImag, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double real = minReal + (maxReal - minReal) * x / (width - 1);
        double imag = minImag + (maxImag - minImag) * y / (height - 1);
        
        int value = mandelbrot(real, imag);
        image[y * width + x] = value;
    }
}

int main() {
    int *image;
    int *d_image;

    // Asignar memoria para la imagen en el host
    image = (int *)malloc(WIDTH * HEIGHT * sizeof(int));

    // Asignar memoria para la imagen en el device (GPU)
    cudaMalloc((void **)&d_image, WIDTH * HEIGHT * sizeof(int));

    // Definir los límites del conjunto de Mandelbrot
    double minReal = -2.0, maxReal = 1.0;
    double minImag = -1.5, maxImag = 1.5;

    // Definir el tamaño de los bloques y la malla de hilos
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Llamar al kernel
    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_image, minReal, maxReal, minImag, maxImag, WIDTH, HEIGHT);

    // Copiar el resultado de la GPU al host
    cudaMemcpy(image, d_image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Inicializar SFML
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Conjunto de Mandelbrot");

    // Crear una textura para mostrar la imagen en pantalla
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);

    // Crear un sprite para mostrar la textura
    sf::Sprite sprite(texture);

    // Crear una imagen SFML para cargar los datos
    sf::Image img;
    img.create(WIDTH, HEIGHT);

    // Variables para imprimir rangos de colores
    int colorRangeLow = MAX_ITER;
    int colorRangeHigh = 0;

    // Convertir los datos de la imagen a formato SFML
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int iter = image[y * WIDTH + x];
            sf::Color color;
            
            if (iter == MAX_ITER) {
                color = sf::Color::Black; // Píxel en el conjunto de Mandelbrot
            } else {
                // Mapear iteraciones a un rango de colores (ahora con más iteraciones)
                int r = (iter * 9) % 256;
                int g = (iter * 9 * 2) % 256;
                int b = (iter * 9 * 3) % 256;

                color = sf::Color(r, g, b);

                // Actualizar los rangos de color
                if (iter < colorRangeLow) colorRangeLow = iter;
                if (iter > colorRangeHigh) colorRangeHigh = iter;
            }
            img.setPixel(x, y, color);
        }
    }

    // Cargar la imagen en la textura
    texture.update(img);

    // Imprimir rangos de colores en consola
    std::cout << "Rango de iteraciones: " << colorRangeLow << " a " << colorRangeHigh << std::endl;
    std::cout << "Colores aproximados para estos rangos:" << std::endl;
    std::cout << "Mínimo iteraciones: (" 
              << (colorRangeLow * 9) % 256 << ", "
              << (colorRangeLow * 9 * 2) % 256 << ", "
              << (colorRangeLow * 9 * 3) % 256 << ")" << std::endl;
    std::cout << "Máximo iteraciones: (" 
              << (colorRangeHigh * 9) % 256 << ", "
              << (colorRangeHigh * 9 * 2) % 256 << ", "
              << (colorRangeHigh * 9 * 3) % 256 << ")" << std::endl;

    // Bucle principal para mostrar la ventana
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    // Liberar memoria
    free(image);
    cudaFree(d_image);

    return 0;
}

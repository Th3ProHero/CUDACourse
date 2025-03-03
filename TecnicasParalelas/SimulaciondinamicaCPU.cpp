#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

const int N = 2048; // AUMENTAMOS la resolución
const float dt = 0.05f;
const int iteraciones = 1000; // MÁS iteraciones

// 🔹 NUEVA FUNCIÓN: Simula vorticidad para hacer cálculos más pesados
void vorticidad(std::vector<std::vector<float>> &grid) {
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            float vort = (grid[i+1][j] - grid[i-1][j]) - (grid[i][j+1] - grid[i][j-1]);
            grid[i][j] += vort * 0.01f * sin(grid[i][j]) * cos(grid[i][j]); // Más operaciones
        }
    }
}

// 🔹 DIFUSIÓN: Más iteraciones internas
void difusion(std::vector<std::vector<float>> &grid, float diff) {
    std::vector<std::vector<float>> temp = grid;
    for (int k = 0; k < 30; k++) {  // MÁS iteraciones
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                float laplaciano = temp[i + 1][j] + temp[i - 1][j] + temp[i][j + 1] + temp[i][j - 1] - 4 * temp[i][j];
                grid[i][j] += diff * laplaciano * dt * exp(-grid[i][j] * 0.01f); // MÁS carga computacional
            }
        }
    }
}

// 🔹 ADVECCIÓN: Más cálculos en cada celda
void adveccion(std::vector<std::vector<float>> &grid, std::vector<std::vector<float>> &vel_x, std::vector<std::vector<float>> &vel_y) {
    std::vector<std::vector<float>> temp = grid;
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            int prevX = i - vel_x[i][j] * dt;
            int prevY = j - vel_y[i][j] * dt;
            if (prevX >= 1 && prevX < N - 1 && prevY >= 1 && prevY < N - 1) {
                grid[i][j] = temp[prevX][prevY] * 0.99f + sin(prevX * prevY * 0.01f) * 0.01f;
            }
        }
    }
}

// 🔥 Simulación completa
int main() {
    std::vector<std::vector<float>> grid(N, std::vector<float>(N, 0.0f));
    std::vector<std::vector<float>> vel_x(N, std::vector<float>(N, 1.0f));
    std::vector<std::vector<float>> vel_y(N, std::vector<float>(N, 0.5f));

    grid[N / 2][N / 2] = 100.0f; // Fuente inicial

    auto start = std::chrono::high_resolution_clock::now();

    for (int it = 0; it < iteraciones; it++) {
        difusion(grid, 0.2f);
        adveccion(grid, vel_x, vel_y);
        vorticidad(grid); // Agregamos más cálculos pesados
    }

    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = stop - start;
    std::cout << "🔥 TIEMPO de ejecución en CPU: " << duration.count() << " segundos" << std::endl;

    return 0;
}

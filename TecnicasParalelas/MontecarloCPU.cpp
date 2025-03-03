#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 1410065408// Número de puntos

// Función CPU para calcular la aproximación de PI usando Monte Carlo
void monteCarloCPU(int &count, int n) {
    // Generar puntos aleatorios y contar los puntos dentro del círculo
    for (int i = 0; i < n; ++i) {
        // Generar números aleatorios entre 0 y 1 para las coordenadas (x, y)
        float x = static_cast<float>(rand()) / RAND_MAX;
        float y = static_cast<float>(rand()) / RAND_MAX;

        // Comprobar si el punto cae dentro del círculo unitario
        if (x * x + y * y <= 1.0f) {
            count++;
        }
    }
}

int main() {
    int count = 0;
    
    // Medir el tiempo de ejecución en CPU
    clock_t start = clock();

    std::cout << "Ejecutando en CPU..." << std::endl;
    
    // Ejecutar el cálculo de Monte Carlo en la CPU
    monteCarloCPU(count, N);

    // Calcular la aproximación de PI
    float pi = 4.0f * count / N;

    // Medir el tiempo de ejecución
    clock_t end = clock();
    double elapsed_time = double(end - start) / CLOCKS_PER_SEC;

    // Mostrar los resultados
    std::cout << "Aproximación de PI: " << pi << std::endl;
    std::cout << "Puntos dentro del círculo: " << count << std::endl;
    std::cout << "Tiempo de ejecución en CPU: " << elapsed_time << " segundos" << std::endl;

    return 0;
}

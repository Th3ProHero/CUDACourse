#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Función para realizar el Bucket Sort en la CPU
void bucketSort(int *arr, int n) {
    // Paso 1: Encuentra el valor máximo y mínimo en el arreglo
    int maxVal = *max_element(arr, arr + n);
    int minVal = *min_element(arr, arr + n);

    // Paso 2: Crear buckets
    int bucketCount = 10;  // Número de cubos
    int range = (maxVal - minVal) / bucketCount + 1;
    vector<vector<int>> buckets(bucketCount);

    // Paso 3: Colocar los elementos en los cubos
    for (int i = 0; i < n; i++) {
        int index = (arr[i] - minVal) / range;
        buckets[index].push_back(arr[i]);
    }

    // Paso 4: Ordenar cada cubo y combinar
    int index = 0;
    for (int i = 0; i < bucketCount; i++) {
        // Ordena cada cubo usando el algoritmo de inserción
        if (!buckets[i].empty()) {
            vector<int> &bucket = buckets[i];
            // Ordenar el cubo en la CPU
            sort(bucket.begin(), bucket.end());
            // Copiar los elementos ordenados de nuevo al arreglo
            for (int j = 0; j < bucket.size(); j++) {
                arr[index++] = bucket[j];
            }
        }
    }
}

int main() {
    const int n = 90000000;  // Número de elementos
    int *arr = new int[n];

    // Inicializar el arreglo con valores aleatorios
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 1000;  // Valores aleatorios entre 0 y 999
    }

    // Imprimir el tamaño de los datos
    cout << "Número de elementos: " << n << endl;

    // Imprimir los primeros 10 elementos antes de ordenar
    cout << "Arreglo antes de ordenar (primeros 10 elementos): ";
    for (int i = 0; i < 10; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Medir el tiempo de ejecución del Bucket Sort
    auto start = high_resolution_clock::now();

    // Ejecutar Bucket Sort en la CPU
    bucketSort(arr, n);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // Imprimir el tiempo de ejecución
    cout << "Tiempo de ejecución del Bucket Sort: " << duration.count() << " milisegundos" << endl;

    // Imprimir los primeros 10 elementos después de ordenar
    cout << "Arreglo ordenado (primeros 10 elementos): ";
    for (int i = 0; i < 10; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Liberar la memoria
    delete[] arr;

    return 0;
}

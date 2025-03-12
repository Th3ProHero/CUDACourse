#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Kernel para distribuir los elementos en los cubos
__global__ void distributeElements(int *arr, int *buckets, int n, int bucketCount, int minVal, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int index = (arr[idx] - minVal) / range;
        atomicAdd(&buckets[index], 1);  // Incrementa el contador de elementos en el cubo
    }
}

// Kernel para realizar el Insertion Sort dentro de cada cubo de forma paralela
__global__ void insertionSortKernel(int *arr, int n) {
    int i = threadIdx.x;
    if (i > 0) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

void bucketSort(int *arr, int n) {
    int maxVal = *max_element(arr, arr + n);
    int minVal = *min_element(arr, arr + n);

    int bucketCount = 10;  // Número de cubos
    int range = (maxVal - minVal) / bucketCount + 1;
    vector<int> buckets(bucketCount, 0);

    // Lanzamos el kernel para distribuir los elementos entre los cubos
    int *d_arr, *d_buckets;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_buckets, bucketCount * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buckets, buckets.data(), bucketCount * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    distributeElements<<<blocks, threadsPerBlock>>>(d_arr, d_buckets, n, bucketCount, minVal, range);

    cudaDeviceSynchronize();

    cudaMemcpy(buckets.data(), d_buckets, bucketCount * sizeof(int), cudaMemcpyDeviceToHost);

    // Paso 3: Colocar los elementos en los cubos
    vector<vector<int>> bucketArray(bucketCount);
    for (int i = 0; i < n; i++) {
        int index = (arr[i] - minVal) / range;
        bucketArray[index].push_back(arr[i]);
    }

    // Paso 4: Ordenar los cubos y combinarlos
    for (int i = 0; i < bucketCount; i++) {
        if (!bucketArray[i].empty()) {
            // Ordena el cubo usando Insertion Sort paralelo
            int *bucketData;
            cudaMalloc(&bucketData, bucketArray[i].size() * sizeof(int));
            cudaMemcpy(bucketData, bucketArray[i].data(), bucketArray[i].size() * sizeof(int), cudaMemcpyHostToDevice);

            // Lanzamos el kernel para ordenar el cubo
            insertionSortKernel<<<1, bucketArray[i].size()>>>(bucketData, bucketArray[i].size());

            cudaDeviceSynchronize();

            cudaMemcpy(bucketArray[i].data(), bucketData, bucketArray[i].size() * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(bucketData);
        }
    }

    // Combinamos los elementos ordenados en el arreglo original
    int index = 0;
    for (int i = 0; i < bucketCount; i++) {
        for (int j = 0; j < bucketArray[i].size(); j++) {
            arr[index++] = bucketArray[i][j];
        }
    }

    cudaFree(d_arr);
    cudaFree(d_buckets);
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

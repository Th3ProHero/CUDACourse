#include <iostream>
using namespace std;

// Función para fusionar dos mitades ordenadas en un solo arreglo ordenado
void merge(int arr[], int left, int mid, int right) {
    // Calcular el tamaño de los subarreglos
    int n1 = mid - left + 1;
    int n2 = right - mid;

    // Crear arreglos temporales para almacenar las mitades
    int leftArr[n1], rightArr[n2];

    // Copiar los datos a los arreglos temporales
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
    }

    cout << "\nFusionando: ";
    for (int i = 0; i < n1; i++) {
        cout << leftArr[i] << " ";
    }
    cout << "y ";
    for (int j = 0; j < n2; j++) {
        cout << rightArr[j] << " ";
    }
    cout << endl;

    // Índices para los arreglos temporales
    int i = 0, j = 0, k = left;

    // Fusionar los arreglos temporales en el arreglo original
    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    // Copiar los elementos restantes de leftArr[], si los hay
    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    // Copiar los elementos restantes de rightArr[], si los hay
    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }

    cout << "Resultado de la fusión: ";
    for (int i = left; i <= right; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// Función para implementar Merge Sort
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        // Encuentra el punto medio
        int mid = left + (right - left) / 2;
        cout << "Dividiendo: ";
        for (int i = left; i <= right; i++) {
            cout << arr[i] << " ";
        }
        cout << "en dos mitades: ";

        // Ordenar la primera y segunda mitad
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        // Fusionar las dos mitades ordenadas
        merge(arr, left, mid, right);
    }
}

int main() {
    int arr[] = { 12, 11, 13, 5, 6, 7 };
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "Arreglo original:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Llamar a la función mergeSort
    mergeSort(arr, 0, n - 1);

    cout << "Arreglo ordenado:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

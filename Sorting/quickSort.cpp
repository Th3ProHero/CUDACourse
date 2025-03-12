#include <iostream>
using namespace std;

// Función para realizar el intercambio de dos elementos
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Función para particionar el arreglo y obtener el índice del pivote
int partition(int arr[], int low, int high) {
    int pivot = arr[high];  // Elegimos el pivote como el último elemento
    int i = (low - 1);  // Índice del menor elemento
    cout << "\nPivote: " << pivot << endl;

    for (int j = low; j < high; j++) {
        // Si el elemento actual es menor o igual al pivote
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    
    // Impresión de depuración para ver el estado después del particionado
    cout << "Estado del arreglo después del particionado: ";
    for (int i = low; i <= high; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return (i + 1);  // Retorna la posición del pivote
}

// Función para implementar el Quick Sort recursivamente
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        // Encuentra el índice de partición
        int pi = partition(arr, low, high);

        // Recursión sobre las dos mitades
        quickSort(arr, low, pi - 1);  // Parte izquierda
        quickSort(arr, pi + 1, high); // Parte derecha
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

    // Llamar a la función quickSort
    quickSort(arr, 0, n - 1);

    cout << "Arreglo ordenado:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

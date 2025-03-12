#include <iostream>
using namespace std;

void bubbleSort(int arr[], int n) {
    // Bucle para recorrer todo el arreglo
    for (int i = 0; i < n - 1; i++) {
        // Bucle para comparar elementos adyacentes
        for (int j = 0; j < n - i - 1; j++) {
            cout << "Comparando arr[" << j << "] = " << arr[j] << " con arr[" << j + 1 << "] = " << arr[j + 1] << endl;
            
            // Si el elemento actual es mayor que el siguiente, los intercambiamos
            if (arr[j] > arr[j + 1]) {
                cout << "Intercambiando arr[" << j << "] = " << arr[j] << " con arr[" << j + 1 << "] = " << arr[j + 1] << endl;
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int arr[] = { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    int n = sizeof(arr) / sizeof(arr[0]);

    cout << "Arreglo original:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    // Llamar a la función bubbleSort
    bubbleSort(arr, n);

    // Mostrar el arreglo ordenado
    cout << "Arreglo ordenado:" << endl;
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;

    return 0;
}

import time
import random
import matplotlib.pyplot as plt

# Algoritmos de ordenamiento

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def bucket_sort(arr):
    # Determinar el número de cubetas
    num_buckets = 10
    max_value = max(arr)
    min_value = min(arr)
    bucket_range = (max_value - min_value) / num_buckets

    # Crear las cubetas
    buckets = [[] for _ in range(num_buckets)]

    # Distribuir los elementos en las cubetas
    for num in arr:
        index = int((num - min_value) // bucket_range)
        if index >= num_buckets:
            index = num_buckets - 1
        buckets[index].append(num)

    # Ordenar cada cubeta (usando insertion sort)
    for bucket in buckets:
        insertion_sort(bucket)

    # Concatenar las cubetas ordenadas
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)

    return sorted_arr

# Función para medir el tiempo de ejecución de un algoritmo
def measure_time(sort_function, arr):
    start_time = time.time()
    if sort_function == bucket_sort:
        sorted_arr = sort_function(arr)
    else:
        sort_function(arr)
    end_time = time.time()
    return end_time - start_time

# Generar una lista de datos aleatorios
data_size = 100000  # Aumentamos el tamaño de los datos
random_data = [random.randint(0, 100000) for _ in range(data_size)]

# Algoritmos a comparar
algorithms = {
    "Bubble Sort": bubble_sort,
    "Selection Sort": selection_sort,
    "Insertion Sort": insertion_sort,
    "Merge Sort": merge_sort,
    "Quick Sort": quick_sort,
    "Bucket Sort": bucket_sort
}

# Listas para almacenar los resultados
algorithm_names = []
execution_times = []

# Medir el tiempo de cada algoritmo
for name, algorithm in algorithms.items():
    data_copy = random_data.copy()  # Copia de los datos para cada algoritmo
    elapsed_time = measure_time(algorithm, data_copy)
    algorithm_names.append(name)
    execution_times.append(elapsed_time)
    print(f"{name}: {elapsed_time:.6f} segundos")

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(algorithm_names, execution_times, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
plt.xlabel('Algoritmos')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Comparación de Tiempos de Algoritmos de Ordenamiento')
plt.show()
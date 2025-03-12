from numba import cuda
import numpy as np
import math
import time  # Para medir el tiempo

@cuda.jit
def merge_sort_kernel(arr, temp, width, n):
    idx = cuda.grid(1)
    start = idx * width * 2
    middle = start + width
    end = min(start + width * 2, n)

    if middle >= n:
        return

    i, j, k = start, middle, start
    while i < middle and j < end:
        if arr[i] < arr[j]:
            temp[k] = arr[i]
            i += 1
        else:
            temp[k] = arr[j]
            j += 1
        k += 1

    while i < middle:
        temp[k] = arr[i]
        i += 1
        k += 1

    while j < end:
        temp[k] = arr[j]
        j += 1
        k += 1

    for i in range(start, end):
        arr[i] = temp[i]

def merge_sort_gpu(arr):
    n = len(arr)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    d_arr = cuda.to_device(arr)
    d_temp = cuda.device_array_like(arr)

    width = 1
    while width < n:
        merge_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_temp, width, n)
        width *= 2

    d_arr.copy_to_host(arr)

# 🔥 Generar un array aleatorio grande
N = 1_000_000  # 🚀 1 millón de elementos
arr_gpu = np.random.randint(0, 1000000, N, dtype=np.int32)
arr_cpu = arr_gpu.copy()

# ⏳ Medir tiempo en GPU
start_gpu = time.perf_counter()
merge_sort_gpu(arr_gpu)
end_gpu = time.perf_counter()

# ⏳ Medir tiempo en CPU (NumPy usa QuickSort optimizado en CPU)
start_cpu = time.perf_counter()
arr_cpu.sort()
end_cpu = time.perf_counter()

# 📌 Mostrar resultados
print(f"✅ Ordenamiento en GPU terminado en {end_gpu - start_gpu:.6f} segundos")
print(f"✅ Ordenamiento en CPU terminado en {end_cpu - start_cpu:.6f} segundos")
print("Primeros 10 elementos ordenados en GPU:", arr_gpu[:10])
print("Primeros 10 elementos ordenados en CPU:", arr_cpu[:10])

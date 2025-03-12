from numba import cuda
import numpy as np
import time

@cuda.jit
def bitonic_sort_kernel(arr, stage, step):
    idx = cuda.grid(1)
    j = idx ^ step  # Operación XOR para encontrar el par con el que se compara

    if j > idx:
        asc = (idx & stage) == 0  # Determina si es ascendente o descendente
        if (asc and arr[idx] > arr[j]) or (not asc and arr[idx] < arr[j]):
            arr[idx], arr[j] = arr[j], arr[idx]  # Intercambio de valores

def bitonic_sort_gpu(arr):
    n = len(arr)
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    d_arr = cuda.to_device(arr)

    stage = 2
    while stage <= n:
        step = stage // 2
        while step > 0:
            bitonic_sort_kernel[blocks_per_grid, threads_per_block](d_arr, stage, step)
            step //= 2
        stage *= 2

    d_arr.copy_to_host(arr)

# 🔥 Generar un array grande (potencia de 2 para Bitonic Sort)
N = 1_048_576  # 🚀 2^20 elementos (1 millón aprox)
arr_gpu = np.random.randint(0, 90000000, N, dtype=np.int32)
arr_cpu = arr_gpu.copy()

# ⏳ Medir tiempo en GPU
start_gpu = time.perf_counter()
bitonic_sort_gpu(arr_gpu)
end_gpu = time.perf_counter()

# ⏳ Medir tiempo en CPU (NumPy usa QuickSort optimizado)
start_cpu = time.perf_counter()
arr_cpu.sort()
end_cpu = time.perf_counter()

# 📌 Mostrar resultados
print(f"✅ Bitonic Sort en GPU: {end_gpu - start_gpu:.6f} segundos")
print(f"✅ QuickSort en CPU: {end_cpu - start_cpu:.6f} segundos")
print("Primeros 10 elementos ordenados en GPU:", arr_gpu[:10])
print("Primeros 10 elementos ordenados en CPU:", arr_cpu[:10])

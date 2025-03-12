from numba import cuda
import numpy as np
import time

@cuda.jit
def radix_sort_kernel(arr, temp, exp, n):
    idx = cuda.grid(1)
    if idx >= n:
        return

    # Obtener el dígito correspondiente
    digit = (arr[idx] // exp) % 10

    # Contar la cantidad de elementos con el mismo dígito
    cuda.atomic.add(temp, digit, 1)

@cuda.jit
def prefix_sum(temp):
    tid = cuda.threadIdx.x
    for d in range(1, 10):
        if tid >= d:
            temp[tid] += temp[tid - d]
        cuda.syncthreads()

@cuda.jit
def reorder_kernel(arr, output, temp, exp, n):
    idx = cuda.grid(1)
    if idx >= n:
        return

    digit = (arr[idx] // exp) % 10
    pos = cuda.atomic.add(temp, digit, -1) - 1
    output[pos] = arr[idx]

def radix_sort_gpu(arr):
    n = len(arr)
    threads_per_block = 1024  # Más hilos para mayor eficiencia
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    d_arr = cuda.to_device(arr)
    d_output = cuda.device_array_like(arr)
    d_temp = cuda.device_array(10, dtype=np.int32)  # Array de conteo de dígitos

    max_val = arr.max()
    exp = 1

    while max_val // exp > 0:
        cuda.synchronize()
        d_temp.copy_to_device(np.zeros(10, dtype=np.int32))  # Reset seguro de memoria

        radix_sort_kernel[blocks_per_grid, threads_per_block](d_arr, d_temp, exp, n)
        cuda.synchronize()

        prefix_sum[1, 10](d_temp)
        cuda.synchronize()

        reorder_kernel[blocks_per_grid, threads_per_block](d_arr, d_output, d_temp, exp, n)
        cuda.synchronize()

        d_arr.copy_to_device(d_output)
        exp *= 10

    d_arr.copy_to_host(arr)

# 🔥 Generar 10 millones de elementos (reduje a 10M para evitar OOM)
N = 10_000_000
arr_gpu = np.random.randint(0, 1_000_000, N, dtype=np.int32)
arr_cpu = arr_gpu.copy()

# ⏳ Medir tiempo en GPU
start_gpu = time.perf_counter()
radix_sort_gpu(arr_gpu)
end_gpu = time.perf_counter()

# ⏳ Medir tiempo en CPU
start_cpu = time.perf_counter()
arr_cpu.sort()
end_cpu = time.perf_counter()

# 📌 Mostrar resultados
print(f"✅ Radix Sort en GPU: {end_gpu - start_gpu:.6f} segundos")
print(f"✅ QuickSort en CPU: {end_cpu - start_cpu:.6f} segundos")
print("Primeros 10 elementos ordenados en GPU:", arr_gpu[:10])
print("Primeros 10 elementos ordenados en CPU:", arr_cpu[:10])

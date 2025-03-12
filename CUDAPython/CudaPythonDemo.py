from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = x[idx] + y[idx]

# Aumentar el tamaño del array
N = 100000  # Más datos para mejor uso de la GPU
x = np.arange(N, dtype=np.float32)
y = np.arange(N, dtype=np.float32)
out = np.zeros_like(x)

# Transferencia a GPU
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(out)

# Configuración óptima
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

print(f"Blocks per grid: {blocks_per_grid}, Threads per block: {threads_per_block}")

# Lanzar kernel
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)

# Copiar resultado de vuelta a CPU
d_out.copy_to_host(out)

print(out[:10])  # Imprimir solo los primeros 10 elementos

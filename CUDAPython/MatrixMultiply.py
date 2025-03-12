from numba import cuda
import numpy as np
import time

# Kernel de multiplicación de matrices
@cuda.jit
def matrix_multiply_kernel(A, B, C, N):
    row, col = cuda.grid(2)
    if row < N and col < N:
        temp = 0.0
        for k in range(N):
            temp += A[row, k] * B[k, col]
        C[row, col] = temp

# Kernel para sumar matrices
@cuda.jit
def matrix_add_kernel(A, B, C, N):
    row, col = cuda.grid(2)
    if row < N and col < N:
        C[row, col] = A[row, col] + B[row, col]

# Kernel para transponer matrices
@cuda.jit
def matrix_transpose_kernel(A, B, N):
    row, col = cuda.grid(2)
    if row < N and col < N:
        B[col, row] = A[row, col]

# Realizar operaciones en la GPU
def matrix_operations_gpu(A, B):
    N = A.shape[0]
    threads_per_block = (32, 32)  # Aumentar hilos por bloque para mayor paralelismo
    blocks_per_grid = (int(np.ceil(N / threads_per_block[0])), int(np.ceil(N / threads_per_block[1])))

    # Copiar las matrices a la memoria de la GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C_multiply = cuda.device_array((N, N), dtype=np.float32)
    d_C_add = cuda.device_array((N, N), dtype=np.float32)
    d_C_transpose = cuda.device_array((N, N), dtype=np.float32)

    # Realizar la multiplicación de matrices
    matrix_multiply_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C_multiply, N)

    # Realizar la suma de matrices
    matrix_add_kernel[blocks_per_grid, threads_per_block](d_A, d_B, d_C_add, N)

    # Realizar la transposición de la matriz
    matrix_transpose_kernel[blocks_per_grid, threads_per_block](d_A, d_C_transpose, N)

    # Copiar los resultados de vuelta a la CPU
    C_multiply = d_C_multiply.copy_to_host()
    C_add = d_C_add.copy_to_host()
    C_transpose = d_C_transpose.copy_to_host()

    return C_multiply, C_add, C_transpose

# Generar matrices grandes con coma flotante (8192x8192)
N = 8192
A = np.random.random((N, N)).astype(np.float32)  # Matrices de tipo float32
B = np.random.random((N, N)).astype(np.float32)

# Medir tiempo de ejecución en GPU
start_gpu = time.perf_counter()
C_multiply_gpu, C_add_gpu, C_transpose_gpu = matrix_operations_gpu(A, B)
end_gpu = time.perf_counter()

# Comparar con CPU
start_cpu = time.perf_counter()
C_multiply_cpu = np.dot(A, B)
C_add_cpu = np.add(A, B)
C_transpose_cpu = np.transpose(A)
end_cpu = time.perf_counter()

# Mostrar resultados
print(f"Operaciones en GPU: {end_gpu - start_gpu:.6f} segundos")
print(f"Operaciones en CPU: {end_cpu - start_cpu:.6f} segundos")
print("Primeros 5 elementos de la multiplicación en GPU vs CPU:")
print(C_multiply_gpu[:5, :5], C_multiply_cpu[:5, :5])
print("Primeros 5 elementos de la suma en GPU vs CPU:")
print(C_add_gpu[:5, :5], C_add_cpu[:5, :5])
print("Primeros 5 elementos de la transposición en GPU vs CPU:")
print(C_transpose_gpu[:5, :5], C_transpose_cpu[:5, :5])

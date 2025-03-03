import numpy as np
import matplotlib.pyplot as plt

# Valores de N (tamaño de los datos)
N_values = np.logspace(2, 4, num=100, dtype=int)  # Generamos valores de N de 100 a 10,000

# Simulación de tiempos para CPU (basado en los valores proporcionados)
# Suponiendo que el tiempo en CPU crece de manera no lineal (aproximadamente cuadrático o cúbico con N)
tiempo_cpu_sim = 1e-3 * (N_values ** 2)  # Ajuste aproximado

# Simulación de tiempos para GPU (basado en los valores proporcionados)
# Suponiendo que el tiempo en GPU crece de manera más moderada (aproximadamente lineal con N)
tiempo_gpu_sim = 0.1 * N_values  # Ajuste aproximado

# Crear la figura y los ejes
plt.figure(figsize=(10, 6))

# Graficar los tiempos simulados
plt.plot(N_values, tiempo_cpu_sim, marker='o', linestyle='-', color='blue', label='CPU Simulado')
plt.plot(N_values, tiempo_gpu_sim, marker='o', linestyle='-', color='orange', label='GPU Simulado')

# Configurar los ejes
plt.xscale('log')  # Usar escala logarítmica para el eje X
plt.yscale('log')  # Usar escala logarítmica para el eje Y
plt.xlabel('Tamaño N (log)')
plt.ylabel('Tiempo (log segs)')
plt.title('Simulación de Tiempos de CPU y GPU para Grandes Valores de N')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Mostrar la gráfica
plt.show()

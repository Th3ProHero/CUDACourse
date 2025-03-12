import matplotlib.pyplot as plt

# Datos de ejemplo (reemplaza con tus propios datos)
sizes = [512, 1024, 2048, 4096, 8192]  # Tamaños de las matrices
gpu_times = [0.1, 0.5, 2.0, 8.0, 32.0]  # Tiempos en GPU (segundos)
cpu_times = [10, 80, 640, 5120, 40960]  # Tiempos en CPU (segundos)

# Crear la gráfica
plt.figure(figsize=(10, 6))

# Graficar los datos
plt.plot(sizes, gpu_times, 'bo-', label='GPU')
plt.plot(sizes, cpu_times, 'ro-', label='CPU')

# Añadir etiquetas y título
plt.xlabel('Tamaño de la matriz (N x N)')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Comparación de tiempos de ejecución en CPU y GPU (Escala lineal)')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
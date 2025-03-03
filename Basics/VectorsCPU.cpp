import psutil
import GPUtil
import os
import time
from tabulate import tabulate

def get_cpu_info():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def get_ram_info():
    ram_info = psutil.virtual_memory()
    return ram_info.percent

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    gpu_info_list = []
    for gpu in gpus:
        gpu_info = {
            'name': gpu.name,
            'load': gpu.load * 100,
            'memoryFree': gpu.memoryFree,
            'memoryUsed': gpu.memoryUsed,
            'memoryTotal': gpu.memoryTotal,
            'temperature': gpu.temperature
        }
        gpu_info_list.append(gpu_info)
    return gpu_info_list

def print_system_stats():
    os.system('cls' if os.name == 'nt' else 'clear')
    cpu_usage = get_cpu_info()
    ram_usage = get_ram_info()
    gpu_stats = get_gpu_info()

    table_data = [
        ['CPU', f'{cpu_usage}%', '', '', ''],
        ['RAM', f'{ram_usage}%', '', '', '']
    ]

    for idx, gpu in enumerate(gpu_stats):
        table_data.append([f'GPU {idx+1} ({gpu["name"]})', f'{gpu["load"]}%', f'{gpu["memoryFree"]}MB', f'{gpu["memoryUsed"]}MB', f'{gpu["temperature"]} °C'])

    print(tabulate(table_data, headers=['Componente', 'Uso', 'Memoria Libre', 'Memoria Usada', 'Temperatura'], tablefmt='grid'))

if __name__ == "__main__":
    while True:
        print_system_stats()
        time.sleep(1)  # Espera 1 segundo antes de actualizar

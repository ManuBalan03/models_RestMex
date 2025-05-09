import os
import random
from pathlib import Path

def dividir_dataset(nombre_archivo):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets_fasttext'))
    ruta_archivo = os.path.join(base_dir, nombre_archivo)

    if not os.path.exists(ruta_archivo):
        print(f"❌ El archivo {nombre_archivo} no existe.")
        return

    # Leer todas las líneas
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()

    # Mezclar aleatoriamente
    random.shuffle(lineas)

    # Calcular corte 80/20
    corte = int(len(lineas) * 0.8)
    train, test = lineas[:corte], lineas[corte:]

    # Crear carpeta de salida
    carpeta_salida = os.path.join(base_dir, 'divididos')
    os.makedirs(carpeta_salida, exist_ok=True)

    # Guardar los archivos
    base_nombre = nombre_archivo.replace('.txt', '')
    with open(os.path.join(carpeta_salida, f'{base_nombre}_train.txt'), 'w', encoding='utf-8') as f_train:
        f_train.writelines(train)

    with open(os.path.join(carpeta_salida, f'{base_nombre}_test.txt'), 'w', encoding='utf-8') as f_test:
        f_test.writelines(test)

    print(f"✅ {nombre_archivo} dividido en {len(train)} para entrenamiento y {len(test)} para prueba.")

if __name__ == "__main__":
    archivos = ['tipo1.txt', 'polaridad1.txt', 'region_ciudad1.txt']
    for archivo in archivos:
        dividir_dataset(archivo)

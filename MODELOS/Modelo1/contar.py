from collections import Counter
import os
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets_fasttext'))
ruta_archivo = os.path.join(base_dir, 'region_ciudad_aumentado.txt')
# ruta_archivo = os.path.join(base_dir, 'tipo.txt')

conteo_etiquetas = Counter()
total_datos = 0

with open(ruta_archivo, 'r', encoding='utf-8') as f:
    for linea in f:
        if linea.startswith('__label__'):
            etiqueta = linea.split()[0]
            conteo_etiquetas[etiqueta] += 1
            total_datos += 1

# Mostrar resultados
print("Etiquetas encontradas y sus cantidades:")
for etiqueta, cantidad in conteo_etiquetas.items():
    porcentaje = (cantidad / total_datos) * 100
    print(f"{etiqueta}: {cantidad} ({porcentaje:.2f}%)")

print(f"\nTotal de etiquetas diferentes: {len(conteo_etiquetas)}")
print(f"Total de datos: {total_datos}")

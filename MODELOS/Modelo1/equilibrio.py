import os
import json
import random
from collections import Counter
from nltk.tokenize import word_tokenize
from itertools import combinations
from tqdm import tqdm 
from difflib import SequenceMatcher

# Rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets_fasttext/divididos'))
ruta_archivo = os.path.join(base_dir, 'tipo1_train.txt')
ruta_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), '../JSON'))
ruta_JsonTipo = os.path.join(ruta_JSON, 'TIPOS.json')

# Cargar JSON de categorías
with open(ruta_JsonTipo, 'r', encoding='utf-8') as json_file:
    categorias_json = json.load(json_file)

# Contar etiquetas y recolectar líneas
conteo_etiquetas = Counter()
lineas_existentes = []

with open(ruta_archivo, 'r', encoding='utf-8') as f:
    for linea in f:
        if linea.startswith('__label__'):
            etiqueta = linea.split()[0]
            conteo_etiquetas[etiqueta] += 1
            lineas_existentes.append(linea.strip())

# Calcular el promedio de datos por etiqueta
cantidadDatos = sum(conteo_etiquetas.values())
cantidadEtiquetas = len(conteo_etiquetas)
promedio = cantidadDatos / cantidadEtiquetas

print(f"Promedio por etiqueta: {promedio}")
print(f"Total de etiquetas: {cantidadEtiquetas}")
print(f"Total de datos: {cantidadDatos}")

def jaccard_similarity(frase1, frase2):
    """Calcula la similitud de Jaccard entre dos frases no vacías y válidas"""
    if not isinstance(frase1, str) or not isinstance(frase2, str):
        return 0.0
    frase1 = frase1.strip().lower()
    frase2 = frase2.strip().lower()
    if not frase1 or not frase2:
        return 0.0
    try:
        set1 = set(word_tokenize(frase1))
        set2 = set(word_tokenize(frase2))
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    except Exception as e:
        print(f"Error tokenizando frases:\n  f1: {frase1}\n  f2: {frase2}\n  Error: {e}")
        return 0.0
    
def recortar_frases_al_promedio(etiqueta, frases, promedio):
    """
    Elimina frases aleatorias de una etiqueta si exceden el promedio.
    """
    exceso = len(frases) - int(promedio)
    if exceso > 0:
        frases = random.sample(frases, int(promedio))
    return frases


# Función para comparar similitud difusa
def similitud_difusa(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Buscar frases similares por similitud difusa
def encontrar_frases_similares(etiqueta, frases, umbral=0.75, max_frases=200):
    similares = []
    frases = frases[:max_frases]  # Limitar cantidad para rendimiento
    for i in range(len(frases)):
        for j in range(i + 1, len(frases)):
            sim = similitud_difusa(frases[i], frases[j])
            if sim >= umbral:
                similares.append(frases[i])
                break  # Si ya encontró una parecida, salta a la siguiente frase
    return similares

# Definir función antes de usarla
def generar_frase(palabras, longitud_min=5, longitud_max=10):
    num_palabras = random.randint(longitud_min, longitud_max)
    seleccionadas = random.sample(palabras, min(num_palabras, len(palabras)))
    frase = ' '.join(seleccionadas)
    return ' '.join(word_tokenize(frase))

# Agrupar frases por etiqueta
frases_por_etiqueta = {}
with open(ruta_archivo, 'r', encoding='utf-8') as f:
    for linea in f:
        if linea.startswith('__label__'):
            partes = linea.strip().split(' ', 1)  # Dividir solo en el primer espacio
            etiqueta = partes[0]
            if len(partes) > 1:
                frase = partes[1]
                frases_por_etiqueta.setdefault(etiqueta, []).append(frase)

# Usar la función para encontrar y eliminar frases similares por etiqueta
frases_a_eliminar = set()
# Usar la función para encontrar frases similares por etiqueta
for etiqueta, frases in tqdm(frases_por_etiqueta.items(), desc="Buscando similares"):
    print(f"Etiqueta: {etiqueta} - Total frases válidas: {len(frases)}")

    # Filtrar frases vacías o nulas
    frases = [f for f in frases if isinstance(f, str) and f.strip()]
    similares = encontrar_frases_similares(etiqueta, frases, umbral=0.8)
    print(f"\nEtiqueta: {etiqueta} - Frases similares encontradas: {len(similares)}")
    for frase in similares[:5]:
        print(f" - {frase}")

    
    # Añadir las frases similares al conjunto de frases a eliminar
    for frase in similares:
        frases_a_eliminar.add(f"{etiqueta} {frase}")

# Filtrar las líneas existentes para eliminar las similares
lineas_filtradas = []
for linea in lineas_existentes:
    partes = linea.strip().split(' ', 1)
    if len(partes) > 1:
        etiqueta = partes[0]
        frase = partes[1]
        if f"{etiqueta} {frase}" not in frases_a_eliminar:
            lineas_filtradas.append(linea)
    else:
        lineas_filtradas.append(linea)  # Mantener líneas sin frase

print(f"\nSe eliminaron {len(lineas_existentes) - len(lineas_filtradas)} frases similares.")

# Ahora aumentar los datos después de eliminar las similares
nuevas_lineas = []
for etiqueta_completa, cantidad in conteo_etiquetas.items():
    categoria = etiqueta_completa.replace('__label__', '')
    
    if categoria in categorias_json:
        # Recalcular la cantidad después de la eliminación
        cantidad_actual = sum(1 for linea in lineas_filtradas if linea.startswith(etiqueta_completa))
        diferencia = int(promedio - cantidad_actual)
        
        if diferencia > 0:
            print(f"Generando {diferencia} frases para: {categoria}")
            palabras_categoria = categorias_json[categoria]
            
            for _ in range(diferencia):
                frase_generada = generar_frase(palabras_categoria)
                nuevas_lineas.append(f"__label__{categoria} {frase_generada}")

# Combinar todas las líneas
todas_lineas = lineas_filtradas + nuevas_lineas

# Reagrupar por etiqueta antes de guardar
agrupadas_final = {}
for linea in todas_lineas:
    partes = linea.strip().split(' ', 1)
    if len(partes) == 2:
        etiqueta, frase = partes
        agrupadas_final.setdefault(etiqueta, []).append(frase)

# Recortar si exceden el promedio
lineas_finales = []
for etiqueta, frases in agrupadas_final.items():
    frases_recortadas = recortar_frases_al_promedio(etiqueta, frases, promedio)
    for frase in frases_recortadas:
        lineas_finales.append(f"{etiqueta} {frase}")

# Guardar resultado final
ruta_salida = os.path.join(base_dir, 'region_ciudad_aumentado.txt')
with open(ruta_salida, 'w', encoding='utf-8') as salida:
    for linea in lineas_finales:
        salida.write(linea + '\n')

print(f"\nSe generaron {len(nuevas_lineas)} frases nuevas.")
print(f"Se eliminaron aleatoriamente frases que excedían el promedio.")
print(f"Archivo aumentado guardado en: {ruta_salida}")

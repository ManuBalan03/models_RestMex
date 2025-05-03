import os
import numpy as np
import pandas as pd
import fasttext

# Parche de numpy para fasttext original
original_array = np.array
def patched_array(*args, **kwargs):
    if 'copy' in kwargs and kwargs['copy'] is False:
        kwargs.pop('copy')
        return np.asarray(*args)
    return original_array(*args, **kwargs)
np.array = patched_array

# Rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(base_dir, 'archivos_bin')
excel_path = os.path.join(base_dir, 'Data/Rest-Mex_2025_test.xlsx') 

output_dir = os.path.join(base_dir, 'resultados')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'resultados_modelo1.xlsx')

# Cargar modelos
model = fasttext.load_model(os.path.join(model_dir, 'modelo_EstMun.bin'))
model2 = fasttext.load_model(os.path.join(model_dir, 'modelo_tipo1.bin'))
model3 = fasttext.load_model(os.path.join(model_dir, 'modelo_polaridad.bin'))

# Leer Excel
df = pd.read_excel(excel_path)
df['Review'] = df['Review'].astype(str)

# Listas para almacenar resultados
ciudades = []
probs_ciudad = []
regiones = []
probs_region = []
tipos = []
probs_tipo = []
polaridades = []
probs_polaridad = []

for comentario in df['Review']:
    comentario = comentario.replace('\n', ' ').strip()
    labelsCiudad, probsCiudad = model.predict(comentario)
    labelsRegion, probsRegion = model.predict(comentario)
    labelsTipo, probsTipo = model2.predict(comentario)
    labelsPolaridad, probsPolaridad = model3.predict(comentario)
    regiones.append(labelsCiudad[0].replace("__label__", "").split("-")[0])
    ciudades.append(labelsCiudad[0].replace("__label__", "").split("-")[1])
    probs_ciudad.append(probsCiudad[0])
    probs_region.append(probsRegion[0])
    tipos.append(labelsTipo[0].replace("__label__", ""))
    probs_tipo.append(probsTipo[0])
    polaridades.append(labelsPolaridad[0].replace("__label__", ""))
    probs_polaridad.append(probsPolaridad[0])

# Agregar columnas al DataFrame original
df['Municipio'] = ciudades
df['Probabilidad Municipio'] = probs_ciudad
df['Estado'] = regiones
df['Probabilidad Estado'] = probs_region
df['Categoria'] = tipos
df['Probabilidad Categoria'] = probs_tipo
df['Polaridad'] = polaridades
df['Probabilidad Polaridad'] = probs_polaridad

# Guardar en nuevo Excel
df.to_excel(output_path, index=False)

print(f"✅ Resultados guardados en: {output_path}")
import fasttext
from transformers import pipeline
import pandas as pd
import os
import numpy as np

# Parche para NumPy 2.0+
original_array = np.array
def patched_array(*args, **kwargs):
    if 'copy' in kwargs and kwargs['copy'] is False:
        kwargs.pop('copy')
        return np.asarray(*args)
    return original_array(*args, **kwargs)
np.array = patched_array

# Pipelines
zero_shot_classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
ner = pipeline("ner", model="mrm8488/bert-spanish-cased-finetuned-ner", aggregation_strategy="simple")

# Rutas
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(base_dir, 'archivos_bin')
data_path = os.path.join(base_dir, 'Data/Rest-Mex_2025_test.xlsx')
resultados_dir = os.path.join(base_dir, "resultados")
os.makedirs(resultados_dir, exist_ok=True)
ruta_excel = os.path.join(resultados_dir, "resultados_modelo2.xlsx")

# Cargar modelos FastText
model_ciudad = fasttext.load_model(os.path.join(model_dir, 'modelo_ciudad.bin'))
model_region = fasttext.load_model(os.path.join(model_dir, 'modelo_region.bin'))

# Leer archivo con reviews
df = pd.read_excel(data_path)

# Etiquetas para clasificación
etiquetas = ["Restaurant", "Hotel", "tourist attraction"]

# Resultados
resultados = []

label_map = {
    "1 star": "1",
    "2 stars": "2",
    "3 stars": "3",
    "4 stars": "4",
    "5 stars": "5"
}

# Procesar cada review
for idx, row in df.iterrows():
    texto = str(row['Review']).replace('\n', ' ').strip()


    try:
      # Zero-shot
      zero_shot_result = zero_shot_classifier(texto, etiquetas, multi_label=False)
      categoria_detectada = zero_shot_result['labels'][0]
      confianza_categoria = round(zero_shot_result['scores'][0], 2)

      # Sentiment
      sentiment_result = sentiment_classifier(texto)
      sentiment_label = sentiment_result[0]['label']
      sentiment_score = round(sentiment_result[0]['score'], 2)
      polaridad = label_map.get(sentiment_label, "desconocido")

      # FastText
      ciudad_pred, probsCiudad = model_ciudad.predict(texto)
      region_pred, probsRegion = model_region.predict(texto)
      
      ciudad = ciudad_pred[0].replace('__label__', '')
      region = region_pred[0].replace('__label__', '')

      resultados.append({
          "ID": row.get('ID', idx),  # Si tienes una columna "ID"
          "Title": row.get('Title', ''),
          "Review": texto,
           "Municipio": ciudad,
          "Probabilidad Municipio": round(probsCiudad[0], 2),
          "Estado": region,
          "Prob_Estado": round(probsRegion[0], 2),
          "Categoría": categoria_detectada,
          "Probabilidad Categoría": confianza_categoria,
          "Polaridad": polaridad,
          "Puntaje polaridad": sentiment_score,
        
      })

    except Exception as e:
        print(f"⚠️ Error en fila {idx}: {e}")

# Crear y guardar DataFrame final
df_resultados = pd.DataFrame(resultados)
df_resultados.to_excel(ruta_excel, index=False)

print(f"\n✅ Resultados guardados correctamente en: {ruta_excel}")

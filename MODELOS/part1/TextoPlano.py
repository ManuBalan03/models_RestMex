import pandas as pd
import re
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from spanlp.domain.strategies import NumbersToVowelsInLowerCase, NumbersToConsonantsInLowerCase, Preprocessing
import os

# Descargar recursos NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# spaCy para español
try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("Instalando modelo de spaCy para español...")
    import os
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")

spanish_stopwords = set(word.lower() for word in stopwords.words('spanish'))

class Preprocesador:
    @staticmethod
    def limpiar_texto(texto):
        if not isinstance(texto, str):
            return ""

        texto = texto.lower().strip()

        # Reemplazar fechas y números largos con 'd'
        patron = r'\b\d{2}/\d{2}/\d{4}\b|\b\d{4,}\b'
        texto = re.sub(patron, lambda m: 'd' * len(m.group()), texto)

        # Reemplazo con estrategias
        strategies = [NumbersToVowelsInLowerCase(), NumbersToConsonantsInLowerCase()]
        texto = Preprocessing().clean(data=texto, clean_strategies=strategies)

        # Eliminar acentos y caracteres especiales
        texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
        texto = re.sub(r'[^a-z\s]', '', texto)

        # Tokenizar y filtrar
        tokens = word_tokenize(texto)
        texto_filtrado = [palabra for palabra in tokens if palabra not in spanish_stopwords]

        # Lematización
        doc = nlp(" ".join(texto_filtrado))
        lemas = [token.lemma_ for token in doc]

        return " ".join(lemas)  # Devuelve texto procesado como cadena

def generar_datasets_fasttext():
    # Cargar datos desde Excel
    ruta_excel = r'C:\Users\IGNITER\OneDrive\Documentos\GitHub\models_RestMex\MODELOS\part1\datos.csv'
    df = pd.read_csv(
        ruta_excel,
        names=["Review", "Polarity", "Town", "Region", "Type"],
        skiprows=1
    )
    
    print(f"Datos cargados: {len(df)} filas")
    print(df.columns)
    
    # Crear directorio para archivos de entrenamiento si no existe
    os.makedirs('datasets_fasttext', exist_ok=True)
    
    # Procesar cada texto y generar archivos para cada modelo
    with open('datasets_fasttext/polaridad.txt', 'w', encoding='utf-8') as f_pol, \
         open('datasets_fasttext/tipo.txt', 'w', encoding='utf-8') as f_tipo, \
         open('datasets_fasttext/region.txt', 'w', encoding='utf-8') as f_region, \
         open('datasets_fasttext/ciudad.txt', 'w', encoding='utf-8') as f_ciudad:
        
        for _, row in df.iterrows():
            if pd.notna(row['Review']):
                texto_procesado = Preprocesador.limpiar_texto(row['Review'])
                
                # Solo escribir si el texto procesado no está vacío
                if texto_procesado.strip():
                    # Archivo para polaridad
                    if pd.notna(row['Polarity']):
                        etiqueta_pol = "pos" if int(row['Polarity']) > 0 else "neg"
                        f_pol.write(f"__label__{etiqueta_pol} {texto_procesado}\n")
                    
                    # Archivo para tipo de establecimiento
                    if pd.notna(row['Type']):
                        tipo = row['Type'].lower().replace(' ', '_')
                        f_tipo.write(f"__label__{tipo} {texto_procesado}\n")
                    
                    # Archivo para región
                    if pd.notna(row['Region']):
                        region = row['Region'].lower().replace(' ', '_')
                        f_region.write(f"__label__{region} {texto_procesado}\n")
                    
                    # Archivo para ciudad
                    if pd.notna(row['Town']):
                        ciudad = row['Town'].lower().replace(' ', '_')
                        f_ciudad.write(f"__label__{ciudad} {texto_procesado}\n")
    
    print("✅ Archivos de entrenamiento generados en la carpeta 'datasets_fasttext':")
    print("  - polaridad.txt - Para el modelo de polaridad (positivo/negativo)")
    print("  - tipo.txt - Para el modelo de tipo de establecimiento")
    print("  - region.txt - Para el modelo de región")
    print("  - ciudad.txt - Para el modelo de ciudad")

def entrenar_modelos_fasttext():
    """
    Código para entrenar los modelos FastText usando los archivos generados.
    Este código no se ejecutará automáticamente, pero se incluye como referencia.
    """
    import fasttext
    
    # Parámetros de entrenamiento recomendados
    params = {
        'lr': 0.1,
        'epoch': 25,
        'wordNgrams': 2,
        'dim': 100,
        'minCount': 2
    }
    
    # Entrenar modelos
    print("Entrenando modelo de polaridad...")
    modelo_polaridad = fasttext.train_supervised(
        'datasets_fasttext/polaridad.txt', **params)
    modelo_polaridad.save_model("modelo_polaridad.bin")
    
    print("Entrenando modelo de tipo...")
    modelo_tipo = fasttext.train_supervised(
        'datasets_fasttext/tipo.txt', **params)
    modelo_tipo.save_model("modelo_tipo.bin")
    
    print("Entrenando modelo de región...")
    modelo_region = fasttext.train_supervised(
        'datasets_fasttext/region.txt', **params)
    modelo_region.save_model("modelo_region.bin")
    
    print("Entrenando modelo de ciudad...")
    modelo_ciudad = fasttext.train_supervised(
        'datasets_fasttext/ciudad.txt', **params)
    modelo_ciudad.save_model("modelo_ciudad.bin")
    
    print("✅ Todos los modelos entrenados y guardados correctamente!")

if __name__ == "__main__":
    generar_datasets_fasttext()
    
    # Descomenta esta línea para entrenar los modelos después de generar los datasets
    # entrenar_modelos_fasttext()
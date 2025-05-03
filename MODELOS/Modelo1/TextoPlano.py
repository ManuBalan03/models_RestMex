import fasttext
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
    ruta_excel = r'C:\Users\IGNITER\OneDrive\Documentos\GitHub\models_RestMex\MODELOS\Modelo1\datos.csv'
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
    with open('datasets_fasttext/polaridad.txt', 'w', encoding='utf-8') as f_pol:
        #  open('datasets_fasttext/tipo.txt', 'w', encoding='utf-8') as f_tipo, \
        #  open('datasets_fasttext/region_ciudad.txt', 'w', encoding='utf-8') as f_region_ciudad:
        
        for _, row in df.iterrows():
            if pd.notna(row['Review']):
                texto_procesado = Preprocesador.limpiar_texto(row['Review'])
                
                # Solo escribir si el texto procesado no está vacío
                if texto_procesado.strip():
                    # Archivo para polaridad
                    if pd.notna(row['Polarity']):
                        f_pol.write(f"__label__{int(row['Polarity'])} {texto_procesado}\n")
                    
                    # Archivo para tipo de establecimiento
                    # if pd.notna(row['Type']):
                    #     tipo = row['Type'].lower().replace(' ', '_')
                    #     f_tipo.write(f"__label__{tipo} {texto_procesado}\n")
                    
                    # # Archivo combinado para región y ciudad
                    # if pd.notna(row['Region']) and pd.notna(row['Town']):
                    #     region = row['Region'].lower().replace(' ', '_')
                    #     ciudad = row['Town'].lower().replace(' ', '_')
                    #     etiqueta_region_ciudad = f"{region}-{ciudad}"
                    #     f_region_ciudad.write(f"__label__{etiqueta_region_ciudad} {texto_procesado}\n")
    
    print("✅ Archivos de entrenamiento generados en la carpeta 'datasets_fasttext':")
    print("  - polaridad.txt - Para el modelo de polaridad (positivo/negativo)")
    print("  - tipo.txt - Para el modelo de tipo de establecimiento")
    print("  - region_ciudad.txt - Para el modelo combinado de región y ciudad")


def entrenar_modelos_fasttext():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_dir = os.path.join(base_dir, 'datasets_fasttext')
    output_dir = os.path.join(base_dir, 'MODELOS', 'archivos_bin')

    params = {
        'lr': 0.1,
        'epoch': 25,
        'wordNgrams': 2,
        'dim': 100,
        'minCount': 2
    }
    print("Entrenando modelo de ciudad-municipio...")
    modelo_ciuMun = fasttext.train_supervised(input=os.path.join(dataset_dir, 'region_ciudad_aumentado.txt'), **params)
    modelo_ciuMun.save_model(os.path.join(output_dir, 'modelo_EstMun.bin'))
    
    print("Entrenando modelo de polaridad...")
    modelo_polaridad = fasttext.train_supervised(input=os.path.join(dataset_dir, 'polaridad_aumentado.txt'), **params)
    modelo_polaridad.save_model(os.path.join(output_dir, 'modelo_polaridad.bin'))

    print("Entrenando modelo de tipo...")
    modelo_tipo = fasttext.train_supervised(input=os.path.join(dataset_dir, 'tipo_aumentado.txt'), **params)
    modelo_tipo.save_model(os.path.join(output_dir, 'modelo_tipo1.bin'))

    print("Todos los modelos entrenados y guardados correctamente!")
    

if __name__ == "__main__":
    # generar_datasets_fasttext()
    
    # Descomenta esta línea para entrenar los modelos después de generar los datasets
    entrenar_modelos_fasttext()
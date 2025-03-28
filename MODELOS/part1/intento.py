import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import spacy


# Descargar recursos de NLTK
# nltk.download()
nltk.download('stopwords')
nltk.download('punkt')

def limpiar_texto(texto):
    # Acentos
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    # Reemplazar números por "d"
    texto = re.sub(r'\d', 'd', texto)
    # Eliminar caracteres especiales
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)  
    # Convertir a minúsculas y eliminar espacios extra
    return texto.lower().strip()

# Texto de prueba
text = "Excelente lugar para comere y pasare una buena noche!!! El servicio es de primera y la comida exquisita!!! 78779"
texto = limpiar_texto(text)

# Cargar stopwords en español y normalizarlas
stop_words = set(limpiar_texto(word) for word in stopwords.words('spanish'))

# Tokenizar palabras
tokens = word_tokenize(texto)

# Eliminar stopwords
texto_filtrado = [palabra for palabra in tokens if palabra not in stop_words]


#-----------------------------------------------
nlp = spacy.load("es_core_news_sm")  # Modelo en español

# Procesar texto
doc = nlp(" ".join(texto_filtrado))  # Corregir aquí, usando " ".join

# Lematización
lematizado = [token.lemma_ for token in doc]
print("Texto lematizado:", " ".join(lematizado))

doc=nlp(" ".join(lematizado))

print("Tokens procesados por spaCy:")
for token in doc:
    print(token.text)

#-----------------------------------------------
from transformers import pipeline

# Cargar modelo de análisis de sentimiento en español
analisis = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")

# Texto a analizar
texto = " ".join(lematizado)

# Analizar sentimiento
resultado = analisis(texto)
print(resultado)


#----------------------------------

# Mostrar resultados
# print("Texto limpio:", texto)
print("Texto sin stopwords:", " ".join(texto_filtrado))
# print("Frases en el texto:", sent_tokenize(texto))
# print("Frecuencia de palabras:", FreqDist(tokens))

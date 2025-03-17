import unicodedata
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
print(nltk.__version__)
# nltk.download()

def limpiar_texto(texto):
    # 1. Normalizar texto y eliminar acentos
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    # 2. Eliminar caracteres especiales, comas, puntos, signos de exclamación, etc.
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)  
    # 3. Convertir a minúsculas y eliminar espacios extra
    return texto.lower().strip()
  
text= "Excelente lugar para comer y pasar una buena noche!!! El servicio es de primera y la comida exquisita!!!"
texto = limpiar_texto(text)
print("Texto limpio:", texto)
tokens=print(word_tokenize(texto))
print(sent_tokenize(texto))
print(FreqDist(word_tokenize(texto)))

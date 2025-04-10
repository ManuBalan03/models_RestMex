from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
import re 
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from spanlp.domain.strategies import NumbersToVowelsInLowerCase, NumbersToConsonantsInLowerCase, Preprocessing

# Descargar recursos NLTK
nltk.download('punkt')
nltk.download('stopwords')

# spaCy para español
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

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def main():
    # Cargar datos desde Excel
    ruta_excel = r'C:\Users\IGNITER\OneDrive\Documentos\GitHub\models_RestMex\MODELOS\part1\datos.csv'
    df = pd.read_csv(
        ruta_excel,
        names=["Review", "Polarity", "Town", "Region", "Type"],
        skiprows=1
    )
    print(f"Datos cargados: {len(df)} filas")
    print(df.columns)
    
    # Procesar textos
    textos_procesados = []
    etiquetas = []
    
    for _, row in df.iterrows():
        if pd.notna(row['Review']) and pd.notna(row['Polarity']):
            texto_procesado = Preprocesador.limpiar_texto(row['Review'])
            textos_procesados.append(texto_procesado)
            
            # Crear etiquetas multicategoría [POS, RESTAURANTE, OAXACA, MERIDA]
            # Esto es un ejemplo, ajusta según tus categorías reales
            es_positivo = 1 if row['Polarity'] > 0 else 0
            es_restaurante = 1 if row['Type'] == 'Restaurant' else 0
            es_oaxaca = 1 if row['Town'] == 'Oaxaca' else 0
            es_merida = 1 if row['Town'] == 'Mérida' else 0
            
            etiquetas.append([es_positivo, es_restaurante, es_oaxaca, es_merida])
    
    print(f"Textos procesados: {len(textos_procesados)}")
    
    # Configurar tokenizer y dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')  # Mejor para español
    max_len = 128
    
    # Crear dataset y dataloader
    batch_size = 16  # Ajusta según tu capacidad de memoria
    dataset = MultiLabelDataset(textos_procesados, etiquetas, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Cargar modelo BERT
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',  # Mejor para español
        num_labels=4  # Ajusta al número de etiquetas
    )
    
    # Configurar entrenamiento
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 5  # Ajusta según necesites
    
    # Entrenamiento
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f"Época {epoch+1}/{num_epochs}, Pérdida: {total_loss/len(dataloader):.4f}")
    
    # Guardar modelo entrenado
    model.save_pretrained('./modelo_restaurantes')
    tokenizer.save_pretrained('./modelo_restaurantes')
    print("¡Modelo guardado correctamente!")

if __name__ == "__main__":
    main()
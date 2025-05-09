import os
import fasttext
import matplotlib.pyplot as plt
from fpdf import FPDF
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import pipeline, AutoTokenizer

script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.abspath(os.path.join(script_dir, '../../MODELOS/archivos_bin/modelo_EstMun.bin'))
print(model_file)
estado_model = fasttext.load_model(os.path.join(model_file))

# Configurar los pipelines con truncación explícita
polaridad_model = pipeline(
    "sentiment-analysis", 
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512
)

categoria_model = pipeline(
    "zero-shot-classification", 
    model="joeddav/xlm-roberta-large-xnli",
    truncation=True,
    max_length=512
)

def cargar_datos_txt(ruta_archivo):
    textos = []
    etiquetas = []
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            partes = linea.strip().split(' ', 1)  # Divide la etiqueta y el comentario
            if len(partes) == 2:
                etiqueta = partes[0].replace('__label__', '')  # Eliminar '__label__'
                texto = partes[1]
                etiquetas.append(etiqueta)
                textos.append(texto)
    return textos, etiquetas

polaridad = os.path.abspath(os.path.join(script_dir, '../../datasets_fasttext/divididos/polaridad1_test.txt'))
tipo = os.path.abspath(os.path.join(script_dir, '../../datasets_fasttext/divididos/tipo1_test.txt'))

X_test_polaridad, y_test_polaridad = cargar_datos_txt(polaridad)
X_test_categoria, y_test_categoria = cargar_datos_txt(tipo)

categorias = ["restaurant", "hotel", "attraction"]

def evaluar_y_guardar(nombre_modelo, y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    texto_reporte = classification_report(y_true, y_pred, digits=4)
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=sorted(list(set(y_true))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(list(set(y_true))))

    disp.plot(xticks_rotation=45)
    imagen_path = f"{nombre_modelo}_confusion.png"
    plt.tight_layout()
    plt.savefig(imagen_path)
    plt.close()
    return texto_reporte, imagen_path

# Procesar los datos en lotes pequeños para evitar superar los límites de memoria
def procesar_en_lotes(modelo, textos, tamano_lote=16):
    resultados = []
    for i in range(0, len(textos), tamano_lote):
        lote = textos[i:i+tamano_lote]
        resultado_lote = modelo(lote)
        resultados.extend(resultado_lote)
    return resultados

# Evaluación modelo polaridad
print("Procesando predicciones de polaridad...")
resultados_polaridad = procesar_en_lotes(polaridad_model, X_test_polaridad)
y_pred_polaridad = [res['label'].split()[0] for res in resultados_polaridad]
reporte_polaridad, img_polaridad = evaluar_y_guardar("polaridad", y_test_polaridad, y_pred_polaridad)

# Evaluación modelo categoría
print("Procesando predicciones de categoría...")
y_pred_categoria = []
for texto in X_test_categoria:
    resultado = categoria_model(texto, candidate_labels=categorias)
    y_pred_categoria.append(resultado['labels'][0])

reporte_categoria, img_categoria = evaluar_y_guardar("categoria", y_test_categoria, y_pred_categoria)

# Crear PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

def agregar_seccion_pdf(titulo, texto, imagen_path):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, titulo, ln=True)
    pdf.set_font("Arial", "", 12)
    for linea in texto.split("\n"):
        pdf.multi_cell(0, 8, linea)
    pdf.image(imagen_path, x=10, w=180)

agregar_seccion_pdf("Evaluación - Polaridad", reporte_polaridad, img_polaridad)
agregar_seccion_pdf("Evaluación - Categoría", reporte_categoria, img_categoria)

pdf.output("reporte_modelos.pdf")
print("PDF generado exitosamente.")
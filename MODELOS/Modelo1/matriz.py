import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fasttext
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

# Asegurarse de que sklearn se importe correctamente con la versión actualizada
try:
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
except ImportError:
    print("Error: Se requiere scikit-learn para ejecutar este script.")
    exit(1)

def manual_confusion_matrix(y_true, y_pred, labels=None):
    """
    Implementación manual de matriz de confusión para evitar problemas con sklearn/numpy
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Crear mapeo de etiqueta a índice
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    # Inicializar matriz con ceros
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    # Llenar la matriz
    for true, pred in zip(y_true, y_pred):
        if true in label_to_index and pred in label_to_index:
            true_idx = label_to_index[true]
            pred_idx = label_to_index[pred]
            cm[true_idx, pred_idx] += 1
    
    return cm

def cargar_datos_test(ruta_archivo):
    """Carga datos de prueba desde un archivo de texto en formato FastText"""
    print(f"Cargando datos de: {ruta_archivo}")
    textos = []
    etiquetas_reales = []
    
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            for linea in f:
                match = re.match(r'__label__(\S+)\s+(.*)', linea.strip())
                if match:
                    etiqueta, texto = match.groups()
                    etiquetas_reales.append(etiqueta)
                    textos.append(texto)
        
        print(f"✅ Datos cargados: {len(textos)} ejemplos con {len(set(etiquetas_reales))} etiquetas únicas")
        return textos, etiquetas_reales
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return [], []

def evaluar_modelo(ruta_modelo, ruta_test, ruta_pdf):
    """
    Evalúa un modelo FastText y genera un informe PDF con métricas.
    """
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: No se encontró el modelo en {ruta_modelo}")
        return
    
    if not os.path.exists(ruta_test):
        print(f"❌ Error: No se encontró el archivo de prueba en {ruta_test}")
        return
    print(f"Cargando modelo: {ruta_modelo}")
    try:
        modelo = fasttext.load_model(ruta_modelo)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    textos, etiquetas_reales = cargar_datos_test(ruta_test)
    if not textos:
        return
    print("Haciendo predicciones...")
    predicciones = []
    
    for texto in textos:
        etiqueta_pred, _ = modelo.predict(texto, k=1)
        etiqueta_limpia = etiqueta_pred[0].replace('__label__', '')
        predicciones.append(etiqueta_limpia)
    
    # Obtener todas las etiquetas únicas
    todas_etiquetas = sorted(list(set(etiquetas_reales + predicciones)))
    print(f"Total de etiquetas únicas: {len(todas_etiquetas)}")
    
    # Calcular métricas
    try:
        # Usar nuestra implementación manual para la matriz de confusión
        print("Calculando matriz de confusión...")
        cm = manual_confusion_matrix(etiquetas_reales, predicciones, labels=todas_etiquetas)
        
        # Calcular accuracy
        print("Calculando métricas generales...")
        acc = accuracy_score(etiquetas_reales, predicciones)
        
        # Calcular métricas por clase (con manejo de errores)
        print("Calculando métricas por clase...")
        try:
            precision = precision_score(etiquetas_reales, predicciones, labels=todas_etiquetas, average='macro', zero_division=0)
            recall = recall_score(etiquetas_reales, predicciones, labels=todas_etiquetas, average='macro', zero_division=0)
            f1 = f1_score(etiquetas_reales, predicciones, labels=todas_etiquetas, average='macro', zero_division=0)
        except Exception as e:
            print(f"⚠️ Advertencia al calcular métricas por clase: {e}")
            # Valores de respaldo
            precision = recall = f1 = 0.0
        
        # Calcular métricas por clase individuales
        metrics_data = []
        for label in todas_etiquetas:
            # Filtrar para esta clase
            y_true_bin = [1 if y == label else 0 for y in etiquetas_reales]
            y_pred_bin = [1 if y == label else 0 for y in predicciones]
            
            # Contar ocurrencias (soporte)
            support = y_true_bin.count(1)
            
            # Calcular métricas si hay ejemplos de esta clase
            if support > 0:
                try:
                    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
                    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                    f1_val = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                except:
                    prec = rec = f1_val = 0.0
            else:
                prec = rec = f1_val = 0.0
            
            metrics_data.append([label, prec, rec, f1_val, support])
        
        # Crear DataFrame de métricas
        metrics_df = pd.DataFrame(metrics_data, columns=['Clase', 'Precisión', 'Recall', 'F1-Score', 'Soporte'])
        
        # Generar PDF
        print(f"Generando PDF en: {ruta_pdf}")
        generar_pdf(ruta_pdf, cm, todas_etiquetas, acc, precision, recall, f1, metrics_df)
        print("✅ PDF generado correctamente")
        
        return True
    
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        return False

def generar_pdf(ruta_pdf, cm, etiquetas, accuracy, precision, recall, f1, metrics_df):
    """Genera un PDF con resultados de evaluación"""
    # Crear directorio para PDF si no existe
    os.makedirs(os.path.dirname(ruta_pdf), exist_ok=True)
    
    with PdfPages(ruta_pdf) as pdf:
        # Página 1: Resumen general
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(0.5, 0.9, "Evaluación del Modelo", fontsize=16, ha='center', weight='bold')
        plt.text(0.5, 0.8, f"Accuracy: {accuracy:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.75, f"Precision (macro): {precision:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.7, f"Recall (macro): {recall:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.65, f"F1-Score (macro): {f1:.4f}", fontsize=14, ha='center')
        plt.text(0.5, 0.55, f"Número de clases: {len(etiquetas)}", fontsize=14, ha='center')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Página 2: Matriz de confusión (si no es demasiado grande)
      # Página 2: Matriz de confusión (siempre mostrar, adaptada a 40 clases)
        plt.figure(figsize=(max(12, len(etiquetas) * 0.3), max(10, len(etiquetas) * 0.3)))

        # Normalizar matriz de confusión
        cm_norm = np.zeros_like(cm, dtype=float)
        for i in range(cm.shape[0]):
            row_sum = np.sum(cm[i])
            if row_sum > 0:
                cm_norm[i] = cm[i] / row_sum

        # Crear heatmap solo con color (sin números si hay muchas clases)
        annotate = cm if len(etiquetas) <= 20 else False

        sns.heatmap(cm_norm, annot=annotate, fmt='d' if annotate is not False else '', cmap='Blues',
                    xticklabels=etiquetas, yticklabels=etiquetas, cbar=True)

        plt.title('Matriz de Confusión', fontsize=14)
        plt.xlabel('Etiqueta Predicha')
        plt.ylabel('Etiqueta Real')

        # Ajustar tamaño de texto
        font_size = 10 if len(etiquetas) <= 20 else 6
        plt.xticks(rotation=90, fontsize=font_size)
        plt.yticks(rotation=0, fontsize=font_size)

        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Página 3: Top 10 métricas por clase
          # Página 3: Métricas por clase
        plt.figure(figsize=(12, len(etiquetas)*0.4 + 3))
        
        tabla_cells = metrics_df.values
        tabla_columns = metrics_df.columns
        
        # Crear tabla con métricas por clase
        tabla = plt.table(
            cellText=tabla_cells,
            colLabels=tabla_columns,
            loc='center',
            cellLoc='center'
        )
        
        # Ajustar estilo de la tabla
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1.2, 1.5)
        
        plt.axis('off')
        plt.title('Métricas por Clase', fontsize=16, pad=20)
        plt.tight_layout()
        
        pdf.savefig()
        plt.close()
        # Gráficos de métricas por clase
        plt.figure(figsize=(12, 8))
        
        # Ordenar por F1-score para mejor visualización
        metrics_sorted = metrics_df.sort_values(by='F1-Score', ascending=False)
        
        # Limitar a 15 clases si hay más (para mejor visualización)
        if len(metrics_sorted) > 15:
            metrics_shown = metrics_sorted.iloc[:15]
            plt.title('Top 15 Clases por F1-Score', fontsize=16)
        else:
            metrics_shown = metrics_sorted
            plt.title('Métricas por Clase', fontsize=16)
        
        # Crear gráfico de barras
        bar_width = 0.25
        index = np.arange(len(metrics_shown))
        
        plt.bar(index, metrics_shown['Precisión'], bar_width, label='Precisión', color='#5DA5DA')
        plt.bar(index + bar_width, metrics_shown['Recall'], bar_width, label='Recall', color='#FAA43A')
        plt.bar(index + 2*bar_width, metrics_shown['F1-Score'], bar_width, label='F1-Score', color='#60BD68')
        
        plt.xlabel('Clase')
        plt.ylabel('Valor')
        plt.xticks(index + bar_width, metrics_shown['Clase'], rotation=90)
        plt.legend()
        plt.tight_layout()
        
        pdf.savefig()
        plt.close()

def buscar_archivos(directorio):
    archivos_test = []
    archivos_modelo = []
    for root, _, files in os.walk(directorio):
        for file in files:
            if file.endswith('.txt') and 'test' in file.lower():
                archivos_test.append(os.path.join(root, file))
            elif file.endswith('.bin') and file.startswith('modelo_'):
                archivos_modelo.append(os.path.join(root, file))
    return archivos_test, archivos_modelo


if __name__ == "__main__":
    # Obtener ruta actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Directorio del script: {script_dir}")
    
    # Definir estructura de directorios base
    # Ajustar estas rutas según la estructura real de tu proyecto
    proyecto_dir = os.path.abspath(os.path.join(script_dir, '../..'))  # Sube dos niveles desde el script
    print(f"Directorio del proyecto: {proyecto_dir}")
    
    # Directorios principales
    datasets_dir = os.path.join(proyecto_dir, 'datasets_fasttext')
    modelos_dir = os.path.join(proyecto_dir, 'modelos')
    reportes_dir = os.path.join(proyecto_dir, 'reportes')
    
    # Asegurarse de que el directorio de reportes existe
    os.makedirs(reportes_dir, exist_ok=True)
    
    print(f"Directorio de datasets: {datasets_dir}")
    print(f"Directorio de modelos: {modelos_dir}")
    print(f"Directorio de reportes: {reportes_dir}")
    
    # Buscar archivos en toda la estructura del proyecto
    print("\n--- Buscando archivos ---")
    # Buscar en la raíz del proyecto y en subdirectorios
    archivos_test, archivos_modelo = buscar_archivos(proyecto_dir)
    
    # Mostrar archivos encontrados
    if archivos_test:
        print("\n=== Archivos de test encontrados ===")
        for i, archivo in enumerate(archivos_test):
            print(f"{i+1}. {archivo}")
    else:
        print("\n❌ No se encontraron archivos de test.")
    
    if archivos_modelo:
        print("\n=== Modelos encontrados ===")
        for i, archivo in enumerate(archivos_modelo):
            print(f"{i+1}. {archivo}")
    else:
        print("\n❌ No se encontraron archivos de modelo.")
    
    # Si tenemos al menos un archivo de cada tipo, proceder
    if archivos_test and archivos_modelo:
        # Usar el primer archivo de cada tipo
        test_file = os.path.join(datasets_dir, 'divididos', 'polaridad1_test.txt')
        model_file = os.path.join(modelos_dir, 'archivos_bin', 'modelo_polaridad.bin')
        
        # Configurar ruta de salida para PDF
        pdf_file = os.path.join(reportes_dir, f"evaluacion_{os.path.basename(model_file).replace('.bin', '')}.pdf")
        
        print(f"\n=== Iniciando evaluación ===")
        print(f"Modelo: {model_file}")
        print(f"Archivo de test: {test_file}")
        print(f"El informe se guardará en: {pdf_file}")
        
        # Evaluar modelo
        evaluar_modelo(model_file, test_file, pdf_file)
    else:
        print("\n❌ No se pudieron encontrar los archivos necesarios para la evaluación.")
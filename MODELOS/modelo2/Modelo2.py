from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

texto = "Este servicio funciona muy rápido"
etiquetas = ["rendimiento", "soporte", "diseño", "facilidad de uso", "bugs"]

resultado = classifier(texto, etiquetas, multi_label=False)

print(resultado)
# Antes de importar fasttext, haz un monkey patch de numpy.array
import numpy as np
import fasttext
original_array = np.array

def patched_array(*args, **kwargs):
    if 'copy' in kwargs and kwargs['copy'] is False:
        kwargs.pop('copy')
        return np.asarray(*args)
    return original_array(*args, **kwargs)

np.array = patched_array

# Cargar el modelo
model = fasttext.load_model("ModeloCiudad.bin")
model1 = fasttext.load_model("ModeloRegion.bin")
model2 = fasttext.load_model("ModeloTipo.bin")

# Comentario de prueba
comentario_nuevo = "la playa estuvo muy bonita mejor que la de progreso las vistas fueron maravillosas "

# Hacer la predicción
labelsCiudad, probsCiudad = model.predict(comentario_nuevo)
labelsRegion, probsRegion = model1.predict(comentario_nuevo)
labelsTipo, probsTipo = model2.predict(comentario_nuevo)

# Mostrar el resultado
print("ciudad")
print("Categoría predicha:", labelsCiudad[0])
print("Probabilidad:", probsCiudad[0])

print("Region")
print("Categoría predicha:", labelsRegion[0])
print("Probabilidad:", probsRegion[0])

print("Tipo")
print("Categoría predicha:", labelsTipo[0])
print("Probabilidad:", probsTipo[0])


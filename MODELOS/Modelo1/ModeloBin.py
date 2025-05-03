import fasttext
import os

ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets_fasttext/ciudad.txt"))
model = fasttext.train_supervised(input=ruta_base)
model.save_model("ModeloCiudad.bin")

ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets_fasttext/region.txt"))
model1 = fasttext.train_supervised(input=ruta_base)
model1.save_model("ModeloRegion.bin")

ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets_fasttext/tipo.txt"))
model2 = fasttext.train_supervised(input=ruta_base)
model2.save_model("ModeloTipo.bin")
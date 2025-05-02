import pandas as pd
import requests
import json
import time

def leer_data(path):
    return pd.read_excel(path)

def preguntar_ollama(prompt, modelo="deepseek-r1:7b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": modelo,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            print(f"Error en respuesta de Ollama: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error conectando a Ollama: {e}")
        return None

# Cargar datos
df = leer_data("Data/Rest-Mex_2025_test.xlsx")

# Agregar columnas vacías para guardar resultados
df["ciudad"] = ""
df["estado"] = ""
df["prob_ciudad"] = ""
df["polaridad"] = ""
df["tipo_lugar"] = ""
df["prob_tipo"] = ""

# Lista de ciudades que puede predecir el modelo y sus estados
ciudades_estado = {
    "Tulum": "Quintana Roo",
    "Isla Mujeres": "Quintana Roo",
    "San Cristóbal de las Casas": "Chiapas",
    "Valladolid": "Yucatán",
    "Bacalar": "Quintana Roo",
    "Palenque": "Chiapas",
    "Sayulita": "Nayarit",
    "Valle de Bravo": "Estado de México",
    "Teotihuacan": "Estado de México",
    "Loreto": "Baja California Sur",
    "Todos Santos": "Baja California Sur",
    "Pátzcuaro": "Michoacán",
    "Taxco": "Guerrero",
    "Tlaquepaque": "Jalisco",
    "Ajijic": "Jalisco",
    "Tequisquiapan": "Querétaro",
    "Metepec": "Estado de México",
    "Tepoztlán": "Morelos",
    "Cholula": "Puebla",
    "Tequila": "Jalisco",
    "Orizaba": "Veracruz",
    "Izamal": "Yucatán",
    "Creel": "Chihuahua",
    "Ixtapan de la Sal": "Estado de México",
    "Zacatlán": "Puebla",
    "Huasca de Ocampo": "Hidalgo",
    "Mazunte": "Oaxaca",
    "Xilitla": "San Luis Potosí",
    "Atlixco": "Puebla",
    "Malinalco": "Estado de México",
    "Bernal": "Querétaro",
    "Tepotzotlán": "Estado de México",
    "Cuetzalan": "Puebla",
    "Chiapa de Corzo": "Chiapas",
    "Parras": "Coahuila",
    "Dolores Hidalgo": "Guanajuato",
    "Coatepec": "Veracruz",
    "Cuatro Ciénegas": "Coahuila",
    "Real de Catorce": "San Luis Potosí",
    "Tapalpa": "Jalisco"
}

ciudades_string = ', '.join(ciudades_estado)

# Iterar sobre cada fila del Excel
for idx, row in df.iterrows():
    titulo = str(row["Title"])
    resena = str(row["Review"])

    prompt = f"""
    Actúa como un analista turístico experto en México. A partir del título y la reseña de un lugar turístico, quiero que estimes:

    1. La ciudad del lugar (de entre las siguientes opciones):
    {ciudades_string}

    2. La probabilidad (de 0 a 1) de que la ciudad sea correcta.
    3. La polaridad general de la reseña, del 1.0 (muy negativo) al 5.0 (muy positivo).
    4. El tipo de lugar: Hotel, Restaurante o Atracción. (OBLIGATORIO SOLO USAR ESTAS 3 OPCIONES)
    5. La probabilidad (de 0 a 1) de que ese tipo de lugar sea correcto.

    IMPORTANTE: No respondas el estado, se inferirá automáticamente según la ciudad seleccionada.

    Responde en este formato JSON:

    ```json
    {{
        "ciudad": "NombreCiudad",
        "prob_ciudad": probabilidad,
        "polaridad": polaridad,
        "tipo_lugar": "Tipo(Hotel, Restaurante o Atracción)",
        "prob_tipo": probabilidad
    }}
    ```

    Información para analizar:
    Título: {titulo}
    Reseña: {resena}
    """

    respuesta = preguntar_ollama(prompt)

    if respuesta:
        try:
            json_inicio = respuesta.find('{')
            json_fin = respuesta.rfind('}') + 1
            json_str = respuesta[json_inicio:json_fin]
            resultado = json.loads(json_str)

            ciudad_predicha = resultado.get("ciudad", "")
            estado_asignado = ciudades_estado.get(ciudad_predicha, "")

            # Asignar valores al DataFrame
            df.at[idx, "ciudad"] = ciudad_predicha
            df.at[idx, "prob_ciudad"] = resultado.get("prob_ciudad", "")
            df.at[idx, "estado"] = estado_asignado
            df.at[idx, "polaridad"] = resultado.get("polaridad", "")
            df.at[idx, "tipo_lugar"] = resultado.get("tipo_lugar", "")
            df.at[idx, "prob_tipo"] = resultado.get("prob_tipo", "")

            print(f"✅ {ciudad_predicha} ({estado_asignado}) - {resultado.get('tipo_lugar', '')} - {resultado.get('polaridad', '')}")

        except Exception as e:
            print(f"❌ Error procesando fila {idx}: {e}")
            continue

    time.sleep(0.5)
    print(f"✔️ Fila {idx + 1}/{len(df)} completada")

# Guardar resultados
df.to_excel("Data/Rest-Mex_2025_results.xlsx", index=False)
print("✅ Resultados guardados en 'Data/Rest-Mex_2025_results.xlsx'")

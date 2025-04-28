from openai import OpenAI
import pandas as pd


def consultar_gpt(title, text):
    client = OpenAI(
        api_key="sk-c5796536a9ac4764bd39a6dde6bbc702",
        base_url="https://api.deepseek.com"
    )

    for intento in range(2):  
        try:
            response = client.chat.completions.create(
                model='deepseek-chat',
                messages=[{
                    "role": "user",
                    "content": "Necesito que hagas una clasificacion (Hotel, Restaurante, Atraccion) del siguiente texto, solo mandame la clasificacion, su titulo es: " + title + " y su texto: " + text
                }]
            )

            print("Categoria ", response)

            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                print(f"Respuesta vacía en el intento {intento+1}")
        
        except Exception as e:
            print(f"Error en el intento {intento+1}: {str(e)}")
        
    return "NONE"


def leer_data(ruta):
    return pd.read_excel(ruta)



df= leer_data("Data/Rest-Mex_2025_test.xlsx")

df['Clasificacion'] = ''

for x in range(50):
    review = df.loc[x, 'Review']
    title = df.loc[x, 'Title']
    
    respuesta = consultar_gpt(title  ,review)

    df.loc[x, 'Clasificacion'] = respuesta

    print(f"Terminado {x} / 100")

df.to_excel("Data/resultado_con_clasificacion.xlsx", index=False)


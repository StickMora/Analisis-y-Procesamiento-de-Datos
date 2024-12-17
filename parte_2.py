# Importar librerías 
import pandas as pd
from datasets import load_dataset
import numpy as np

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]

# Convertir el objeto Dataset a un DataFrame de Pandas
df = pd.DataFrame(data)

# Separar el DataFrame en dos: uno con personas fallecidas y otro con las demás
df_dead = df[df["is_dead"] == 1]
df_alive = df[df["is_dead"] == 0]

# Calcular los promedios de las edades
average_age_dead = df_dead["age"].mean()
average_age_alive = df_alive["age"].mean()


print(f"El promedio de edad de las personas fallecidas es: {average_age_dead:.2f} años")
print(f"El promedio de edad de las personas que sobrevivieron es: {average_age_alive:.2f} años")

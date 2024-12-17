# Importar librerías
from datasets import load_dataset
import numpy as np

# Cargar el dataset
dataset = load_dataset("mstz/heart_failure")

# Acceder a los registros de la partición 'train'
data = dataset["train"]

# Extraer la lista de edades
ages = data["age"]

# Convertir la lista de edades a un arreglo de NumPy
ages_array = np.array(ages)

# Calcular el promedio de edad
average_age = np.mean(ages_array)


print(f"El promedio de edad de los participantes en el estudio es: {average_age:.2f} años")

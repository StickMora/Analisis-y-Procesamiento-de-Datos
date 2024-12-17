# Importar librerías
import pandas as pd
from datasets import load_dataset

# Cargar el dataset y convertirlo a un DataFrame
dataset = load_dataset("mstz/heart_failure")
data = dataset["train"]
df = pd.DataFrame(data)

# Verificar los tipos de datos en el DataFrame
print("Tipos de datos en cada columna:")
print(df.dtypes)

# Asegurarse de que todas las columnas numéricas estén en el formato adecuado
numeric_columns = [
    "age", 
    "creatinine_phosphokinase_concentration_in_blood", 
    "heart_ejection_fraction", 
    "platelets_concentration_in_blood", 
    "serum_creatinine_concentration_in_blood", 
    "serum_sodium_concentration_in_blood", 
    "days_in_study"
]
# Convertir columnas numéricas a tipo float, si es necesario
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Calcular la cantidad de hombres fumadores vs mujeres fumadoras
smokers_gender = df[df["is_smoker"] == 1].groupby("is_male").size()

# Mostrar los resultados
print("\nCantidad de fumadores por género:")
print(f"Hombres fumadores: {smokers_gender.get(1, 0)}")
print(f"Mujeres fumadoras: {smokers_gender.get(0, 0)}")

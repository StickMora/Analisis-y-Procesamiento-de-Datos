import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

# Cargar el dataset desde Hugging Face
dataset = load_dataset("mstz/heart_failure")

# Convertir la partición 'train' del dataset a un DataFrame de Pandas
df = pd.DataFrame(dataset["train"])

# Ver los primeros registros para confirmar la estructura
print(df.head())

def distribucion_edades(df):
    # 1. Histograma de la distribución de edades
    plt.figure(figsize=(10, 6))
    plt.hist(df["age"], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribución de Edades")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Llamar a la función de gráficos
distribucion_edades(df)

def histograma_combinado(df):
    """
    Genera un gráfico combinado de barras agrupadas para hombres y mujeres 
    en las categorías de anémicos, diabéticos, fumadores y muertos.
    """
    # Diccionario con las columnas y sus descripciones
    categories = {
        "has_anaemia": "Anémicos",
        "has_diabetes": "Diabéticos",
        "is_smoker": "Fumadores",
        "is_dead": "Muertos"
    }
    
    # Configurar el gráfico
    bar_width = 0.4
    x_labels = list(categories.values())  # Etiquetas de las categorías
    x_indices = range(len(categories))  # Índices para las barras
    
    # Inicializar los valores de hombres y mujeres
    male_values = []
    female_values = []

    for column in categories.keys():
        # Conteo de hombres y mujeres para cada categoría
        male_count = df[df["is_male"] == 1][column].sum()
        female_count = df[df["is_male"] == 0][column].sum()
        male_values.append(male_count)
        female_values.append(female_count)

    # Crear gráfico de barras agrupadas
    plt.figure(figsize=(10, 6))
    plt.bar(
        [i - bar_width / 2 for i in x_indices],
        male_values,
        width=bar_width,
        label="Hombres",
        color="blue",
        alpha=0.7
    )
    plt.bar(
        [i + bar_width / 2 for i in x_indices],
        female_values,
        width=bar_width,
        label="Mujeres",
        color="red",
        alpha=0.7
    )

    # Configurar el gráfico
    plt.xticks(x_indices, x_labels)
    plt.title("Distribución por Categorías y Género")
    plt.xlabel("Categorías")
    plt.ylabel("Cantidad")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

# Ejemplo
if __name__ == "__main__":
    # Cargar datos desde Hugging Face y convertirlos a DataFrame
    from datasets import load_dataset
    dataset = load_dataset("mstz/heart_failure")
    df = pd.DataFrame(dataset["train"])

    # Generar gráfico combinado
    histograma_combinado(df)

import pandas as pd
import matplotlib.pyplot as plt

def graficar_tortas(dataframe):
    """
    Genera subplots con gráficas de torta para las categorías:
    - Cantidad de anémicos
    - Cantidad de diabéticos
    - Cantidad de fumadores
    - Cantidad de muertos

    Args:
        dataframe (pd.DataFrame): El DataFrame con los datos.
    """
    # Configurar categorías y títulos
    categorias = {
        "has_anaemia": "Anémicos",
        "has_diabetes": "Diabéticos",
        "is_smoker": "Fumadores",
        "is_dead": "Muertos"
    }

    # Crear figura y subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 subplots
    axes = axes.flatten()  # Convertir la matriz de ejes a una lista

    # Generar gráficas de torta
    for i, (columna, titulo) in enumerate(categorias.items()):
        # Calcular proporciones
        conteo = dataframe[columna].value_counts()
        etiquetas = ["No", "Sí"]
        valores = [conteo.get(0, 0), conteo.get(1, 0)]

        # Crear gráfica de torta
        axes[i].pie(
            valores, 
            labels=etiquetas, 
            autopct='%1.1f%%', 
            colors=["skyblue", "lightcoral"], 
            startangle=90, 
            explode=(0, 0.1)
        )
        axes[i].set_title(titulo, fontsize=12)

    # Ajustar diseño
    plt.tight_layout()
    plt.show()

# Ejemplo
if __name__ == "__main__":
    # Cargar datos desde Hugging Face y convertirlos a DataFrame
    from datasets import load_dataset
    dataset = load_dataset("mstz/heart_failure")
    df = pd.DataFrame(dataset["train"])

    # Generar subplots de tortas
    graficar_tortas(df)

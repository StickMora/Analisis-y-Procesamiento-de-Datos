import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def predecir_edades(dataframe):
    """
    Entrena un modelo de regresión lineal para predecir las edades en base a otras características
    y calcula el error cuadrático medio (MSE).

    Args:
        dataframe (pd.DataFrame): El DataFrame original con los datos.

    Returns:
        float: El error cuadrático medio de las predicciones.
    """
    # Crear una copia del dataframe para no modificar el original
    df = dataframe.copy()

    # Eliminar columnas no deseadas
    df = df.drop(columns=["is_dead", "categoria_edad", "age"], errors="ignore")

    # Separar variables independientes (X) y dependientes (y)
    X = df.drop(columns=["age"], errors="ignore")  # Todas las columnas excepto la edad
    y = dataframe["age"]  # Columna objetivo (edades)

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ajustar modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # Predicción de las edades en los datos de prueba
    y_pred = modelo.predict(X_test)

    # Calcular el error cuadrático medio (MSE)
    mse = mean_squared_error(y_test, y_pred)

    print("Edades reales:", y_test.values)
    print("Edades predichas:", y_pred)
    print(f"Error cuadrático medio (MSE): {mse:.2f}")

    return mse


if __name__ == "__main__":
    # Cargar datos desde Hugging Face y convertirlos a DataFrame
    from datasets import load_dataset
    dataset = load_dataset("mstz/heart_failure")
    df = pd.DataFrame(dataset["train"])

    # Agregar columna 'categoria_edad' para simular su eliminación
    df["categoria_edad"] = pd.cut(
        df["age"],
        bins=[0, 12, 19, 39, 59, np.inf],
        labels=["Niño", "Adolescente", "Joven adulto", "Adulto", "Adulto mayor"]
    )

    # Predecir edades y calcular el MSE
    mse = predecir_edades(df)

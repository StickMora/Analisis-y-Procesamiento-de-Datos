import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def clasificacion_arbol_decision(dataframe):
    """
    Ajusta un modelo de clasificación con un árbol de decisión sobre el dataset,
    realiza la partición estratificada, y calcula el accuracy.
    """
    # Eliminar columna 'categoria_edad'
    df = dataframe.drop(columns=["categoria_edad"], errors="ignore")
    
    # Definir características (X) y objetivo (y)
    X = df.drop(columns=["is_dead"], errors="ignore")  # Excluimos la columna objetivo
    y = df["is_dead"]  # Columna objetivo (muerte: 0 o 1)
    
    # Graficar la distribución de clases
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title("Distribución de Clases: Muerte vs No Muerte")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    plt.xticks([0, 1], ["No Muerte (0)", "Muerte (1)"], rotation=0)
    plt.show()

    # Partición estratificada en conjunto de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Ajustar un árbol de decisión
    arbol = DecisionTreeClassifier(random_state=42)
    arbol.fit(X_train, y_train)

    # Predecir sobre el conjunto de test
    y_pred = arbol.predict(X_test)

    # Calcular el accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy del modelo con árbol de decisión: {accuracy:.4f}")

# Ejemplo
if __name__ == "__main__":
    # Cargar datos desde Hugging Face y convertirlos a DataFrame
    from datasets import load_dataset
    dataset = load_dataset("mstz/heart_failure")
    df = pd.DataFrame(dataset["train"])

    # Predecir y evaluar el modelo de clasificación
    clasificacion_arbol_decision(df)

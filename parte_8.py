import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def clasificacion_random_forest(dataframe):
    """
    Ajusta un modelo Random Forest, calcula la matriz de confusión, el F1-Score y el accuracy.
    También permite comparar ambos para evaluar el rendimiento del modelo.
    """
    # Eliminar columna 'categoria_edad'
    df = dataframe.drop(columns=["categoria_edad"], errors="ignore")
    
    # Definir características (X) y objetivo (y)
    X = df.drop(columns=["is_dead"], errors="ignore")  # Excluir la columna objetivo
    y = df["is_dead"]  # Columna objetivo (muerte: 0 o 1)
    
    # Partición estratificada en conjunto de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Ajustar un modelo Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)

    # Predicciones
    y_pred = rf.predict(X_test)

    # Calcular el accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy del modelo Random Forest: {accuracy:.4f}")

    # Calcular el F1-Score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-Score del modelo Random Forest: {f1:.4f}")

    # Calcular y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)

    # Visualizar la matriz de confusión usando ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Muerte", "Muerte"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Random Forest")
    plt.show()

    return accuracy, f1

# Ejemplo
if __name__ == "__main__":
    # Cargar datos desde Hugging Face y convertirlos a DataFrame
    from datasets import load_dataset
    dataset = load_dataset("mstz/heart_failure")
    df = pd.DataFrame(dataset["train"])

    # Predecir y evaluar el modelo de clasificación Random Forest
    accuracy, f1 = clasificacion_random_forest(df)

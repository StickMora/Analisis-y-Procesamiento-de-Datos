import pandas as pd

def limpiar_y_preparar_datos(df):
    """
    Función para limpiar y preparar un DataFrame.
    Realiza:
    1. Verificación y eliminación de valores faltantes.
    2. Eliminación de filas duplicadas.
    3. Detección y eliminación de valores atípicos.
    4. Creación de una nueva columna de categorías de edad.
    5. Guarda el resultado en un archivo CSV.
    
    Args:
        df (pd.DataFrame): El DataFrame a procesar.
        
    Returns:
        pd.DataFrame: El DataFrame limpio y preparado.
    """
    # Verificar y eliminar valores faltantes
    print("Verificando valores faltantes...")
    print(df.isnull().sum())
    df = df.dropna(subset=["Edad", "Email", "Compras"])

    # Verificar y eliminar filas duplicadas
    print("\nVerificando filas duplicadas...")
    print(f"Filas duplicadas antes de eliminar: {df.duplicated().sum()}")
    df = df.drop_duplicates()

    # Detección y eliminación de valores atípicos
    print("\nDetectando valores atípicos en las columnas numéricas...")
    numeric_columns = ["Edad", "Compras", "Ingresos"]
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Valores atípicos en '{col}': {len(outliers)}")
        df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]

    # Crear una columna que categorice por edades
    print("\nCreando columna de categorías de edad...")
    bins = [0, 12, 19, 39, 59, float("inf")]
    labels = ["Niño", "Adolescente", "Joven adulto", "Adulto", "Adulto mayor"]
    df["CategoriaEdad"] = pd.cut(df["Edad"], bins=bins, labels=labels, right=False)

    # Guardar el DataFrame limpio como CSV
    output_file = "datos_limpios.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDatos limpios guardados en: {output_file}")

    return df

# Leer los datos desde el archivo CSV proporcionado
archivo = "datos.csv"
df_original = pd.read_csv(archivo)

# Llamar a la función para limpiar y preparar los datos
df_limpio = limpiar_y_preparar_datos(df_original)

# Mostrar las primeras filas del DataFrame limpio
print("\nPrimeras filas del DataFrame limpio:")
print(df_limpio.head())

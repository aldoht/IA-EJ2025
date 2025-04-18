import pandas as pd

# Carga de los archivos en formato csv
df_train = pd.read_csv('../CSV/written_name_train.csv')
df_test = pd.read_csv('../CSV/written_name_test.csv')
df_validation = pd.read_csv('../CSV/written_name_validation.csv')

# Exploración preliminar: dimensiones
print(f'Cantidad de registros para entrenamiento: {df_train.shape[0]}',
      f'Cantidad de registros para pruebas: {df_test.shape[0]}',
      f'Cantidad de registros para validación: {df_validation.shape[0]}',
      f'Total de registros: {df_train.shape[0] + df_test.shape[0] + df_validation.shape[0]}',
      f'Cantidad de columnas: {df_train.shape[1]}',
      sep='\n')

# Exploración preliminar: primeros registros
print('Estos son los primeros 10 registros del conjunto de entrenamiento:',
      df_train.head(n=10),
      sep='\n')
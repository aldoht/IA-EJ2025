import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import matplotlib.pyplot as plt

# Preview del dataset
file_path = "CSV/"
train_df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "ssarkar445/handwriting-recognitionocr",
  file_path + 'written_name_train.csv'
)
test_df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "ssarkar445/handwriting-recognitionocr",
  file_path + 'written_name_test.csv'
)
validation_df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "ssarkar445/handwriting-recognitionocr",
  file_path + 'written_name_validation.csv'
)

# Set con todos los caracteres de los nombres para entrenamiento
all_chars = set()
for index, value in train_df['IDENTITY'].items():
    for char in str(value):
        all_chars.add(char)
print('Todos los caracteres del conjunto de entrenamiento: \n',
      all_chars)

# Cambiar todos los caracteres por mayusculas
train_df['IDENTITY'] = train_df['IDENTITY'].apply(lambda word: str(word).upper())
test_df['IDENTITY'] = test_df['IDENTITY'].apply(lambda word: str(word).upper())
validation_df['IDENTITY'] = validation_df['IDENTITY'].apply(lambda word: str(word).upper())

# Diccionario con todos los caracteres de los nombres para entrenamiento (en mayusculas)
all_chars = {}
for index, value in train_df['IDENTITY'].items():
    for char in str(value):
        if char in all_chars:
            all_chars[char] += 1
        else:
            all_chars[char] = 1
print('Conteo de todos los caracteres del conjunto de entrenamiento (ya en mayúsculas): \n',
      all_chars)

# Grafica
count = pd.Series(all_chars)
count = count.sort_values(ascending=False)

plt.style.use('ggplot')  # or try 'ggplot', 'seaborn-darkgrid', etc.
fig, ax = plt.subplots(figsize=(10, 6))
count.plot(kind='bar', color='#1f77b4', edgecolor='black', ax=ax)
ax.set_title('Character Count in Tags', fontsize=16, fontweight='bold')
ax.set_xlabel('Characters', fontsize=14)
ax.set_ylabel('Frequency', fontsize=14)
ax.tick_params(axis='x', rotation=0, labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)
plt.tight_layout()
plt.show()

# Cambiar todos los paths a las imagenes
train_df['FILENAME'] = train_df['FILENAME'].map('/train_v2/train/{}'.format)
test_df['FILENAME'] = test_df['FILENAME'].map('/test_v2/test/{}'.format)
validation_df['FILENAME'] = validation_df['FILENAME'].map('/validation_v2/validation/{}'.format)

# Dataset limpio
print('Dataset limpio...',
      'Primeros 5 registros para entrenamiento:',
      train_df.head(),
      'Primeros 5 registros para pruebas:',
      test_df.head(),
      'Primeros 5 registros para validación:',
      validation_df.head(),
      sep='\n')
# Importación de librerías
import math
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Carga del dataset
data = pd.read_csv('./articulos_ml.csv')

# Análisis del dataset
print(data.shape)
print(data.head())
print(data.describe())
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.show()

# Filtrado de datos
filtered_data = data[(data['Word count'] <= 3500) & (data['# Shares'] <= 80000)]
f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values
colored_set = []
for index, row in filtered_data.iterrows():
    if row['Word count'] > 1808:
        colored_set.append('red')
    else:
        colored_set.append('blue')

# Modelo de regresión
X_train = filtered_data[['Word count']].values
y_train = filtered_data['# Shares'].values
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
char = '+' if model.intercept_ >= 0 else '-'
print(f'Function: y = {round(model.coef_[0], 2)}x {char} {round(model.intercept_, 2)}',
      f'Mean Squared Error: {round(mean_squared_error(y_train, y_pred), 2)}',
      f'Variance score: {round(r2_score(y_train, y_pred), 2)}',
      sep='\n')

# Gráfica 2D
plt.scatter(f1, f2, c=colored_set, s=30)
plt.axline((0, model.intercept_), slope=model.coef_[0])
plt.show()

# Predicciones
print(f'Se estiman {math.floor(model.predict([[2000]])[0])} compartidos para un artículo de 2000 palabras.')
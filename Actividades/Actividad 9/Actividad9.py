import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('./articulos_ml.csv')
data.drop(['Title', 'url', 'Elapsed days'], axis=1).hist()
plt.show()

filtered_data = data[(data['Word count'] <= 3500) & data['# Shares'] <= 80000]
colors = ['orange', 'blue']
size = [30, 60]
f1 = filtered_data['Word count'].values
f2 = filtered_data['# Shares'].values
assign = []
for index, row in filtered_data.iterrows():
    if row['Word count'] > 1808:
        assign.append(colors[0])
    else:
        assign.append(colors[1])
plt.scatter(f1, f2, c=assign, s=size[0])
plt.show()
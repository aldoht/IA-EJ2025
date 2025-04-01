import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Data visualization
df = pd.read_csv(r'reviews_sentiment.csv', sep=';')
print(df.head(10),
      df.describe(),
      sep='\n')
df.hist()
plt.show()
print(df.groupby('Star Rating').size())
sb.catplot(data=df, x='Star Rating', kind='count', aspect=3)
plt.show()
sb.catplot(data=df, x='wordcount', kind='count', aspect=3)
plt.show()

# Data preparation
X = df[['wordcount', 'sentimentValue']].values
y = df['Star Rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN model
knn = KNeighborsClassifier(7)
knn.fit(X_train, y_train)
print(f'Accuracy of KNN classifier on training set: {round(knn.score(X_train, y_train), 3)}',
      f'Accuracy of KNN classifier on test set: {round(knn.score(X_test, y_test), 3)}',
      sep='\n')

# Model precision
prediction = knn.predict(X_test)
print(confusion_matrix(y_test, prediction),
      classification_report(y_test, prediction),
      sep='\n')

# Plotting a graph
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])
classifier = KNeighborsClassifier(7, weights='distance')
classifier.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

patch0 = mpatches.Patch(color='#FF0000', label='1')
patch1 = mpatches.Patch(color='#ff9933', label='2')
patch2 = mpatches.Patch(color='#FFFF00', label='3')
patch3 = mpatches.Patch(color='#00ffff', label='4')
patch4 = mpatches.Patch(color='#00FF00', label='5')
plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])
plt.title('5-Class classification (k = 7, weights = distance)')
plt.show()

# Choose the best k
k_range = range(1, 20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20])
plt.show()

# Predictions
print(classifier.predict_proba([[5, 1.0]]))
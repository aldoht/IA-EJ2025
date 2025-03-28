import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import seaborn as sb

df = pd.read_csv('./usuarios_win_mac_lin.csv')
print(df.shape,
      df.head(),
      df.describe(),
      df.groupby('clase').size(),
      sep='\n')

df.drop(['clase'], axis=1).hist()
plt.show()
sb.pairplot(df.dropna(),
            hue='clase',
            height=4,
            vars=['duracion', 'paginas', 'acciones', 'valor'],
            kind='reg')
plt.show()

X = np.array(df.drop(['clase'], axis=1))
y = np.array(df['clase'])
logistic_model = linear_model.LogisticRegression()
logistic_model.fit(X, y)
predictions = logistic_model.predict(X)
print(f'Predictions: {predictions[0:5]}',
      f'Model score: {logistic_model.score(X,y)}',
      sep='\n')

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,
                                                                                y,
                                                                                test_size=0.2,
                                                                                random_state=7)
kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
cv_results = model_selection.cross_val_score(logistic_model,
                                             X_train,
                                             Y_train,
                                             cv=kfold,
                                             scoring='accuracy')
print(f'Logistic regression model with cross validation score: {cv_results.mean()} ({cv_results.std()})')
predictions_cv = logistic_model.predict(X_validation)
print(f'Accuracy: {accuracy_score(Y_validation, predictions_cv)}.')
print(f'Confusion matrix: \n{confusion_matrix(Y_validation, predictions_cv)}')
print(classification_report(Y_validation, predictions_cv))

x_new = pd.DataFrame({'duracion': [10],
                      'paginas': [3],
                      'acciones': [5],
                      'valor': [9]})
print(f'Prediction for new data: {logistic_model.predict(x_new)[0]}')
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')
y = df['Class']
X = df.drop(['Class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

def show_results(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 12))
    sb.heatmap(conf_matrix, xticklabels=True, yticklabels=True, annot=True, fmt='d')
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test, y_pred))

model = RandomForestClassifier(n_estimators=100,
                               bootstrap=True, verbose=2,
                               max_features='sqrt',
                               n_jobs=8,
                               oob_score=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
show_results(y_test, predictions)
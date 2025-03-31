# Library imports
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.model_selection import KFold
from graphviz import render

# Initial analysis
df = pd.read_csv('./artists_billboard_fix3.csv')
print(df.shape,
      df.head(),
      df.groupby('top').size(),
      sep='\n')
sb.catplot(data=df, x='artist_type', kind='count')
plt.show()
sb.catplot(data=df, x='mood', kind='count', aspect=3)
plt.show()
sb.catplot(data=df, x='tempo', hue='top', kind='count')
plt.show()
sb.catplot(data=df, x='genre', kind='count', aspect=3)
plt.show()
sb.catplot(data=df, x='anioNacimiento', kind='count', aspect=3)
plt.show()

# Balancing data
f1 = df['chart_date'].values
f2 = df['durationSeg'].values
a1 = []
a2 = []
for index, row in df.iterrows():
      a1.append('orange' if row['top'] == 0 else 'blue')
      a2.append(60 if row['top'] == 0 else 40)
plt.scatter(f1, f2, c=a1, s=a2)
plt.axis((20030101,20160101,0,600))
plt.show()

# Data preparation
def edad_fix(year: int) -> None | int:
      return None if year == 0 else year
df['anioNacimiento'] = df.apply(lambda x: edad_fix(x['anioNacimiento']), axis=1)
def edad_calc(year: float, when: any) -> None | int:
      return None if year == 0.0 else int(str(when)[:4]) - year
df['edad_en_billboard'] = df.apply(lambda x: edad_calc(x['anioNacimiento'],
                                                       x['chart_date']),
                                   axis=1)
age_avg = df['edad_en_billboard'].mean()
age_std = df['edad_en_billboard'].std()
age_null_count = df['edad_en_billboard'].isnull().sum()
age_null_random_list = np.random.randint(low=age_avg-age_std,
                                         high=age_avg+age_std,
                                         size=age_null_count)
withNullValues = np.isnan(df['edad_en_billboard'])
df.loc[np.isnan(df['edad_en_billboard']), 'edad_en_billboard'] = age_null_random_list
df['edad_en_billboard'] = df['edad_en_billboard'].astype(int)
print(f'Mean age: {round(age_avg, 2)},',
      f'Std deviation age: {round(age_std, 2)},',
      f'Interval for random age: {round(age_avg-age_std, 2)} - {round(age_avg+age_std, 2)}')
f1 = df['edad_en_billboard'].values
f2 = df.index
a3 = []
for index, row in df.iterrows():
      a3.append('green' if withNullValues[index]
                else 'orange' if row['top'] == 0
                else 'blue')
plt.scatter(f1, f2, c=a3, s=30)
plt.axis((15, 50, 0, 500))
plt.show()

# Data mapping
df['moodEncoded'] = df['mood'].map({
      'Energizing': 6,
      'Empowering': 6,
      'Cool': 5,
      'Yearning': 4,
      'Excited': 5,
      'Defiant': 3,
      'Sensual': 2,
      'Gritty': 3,
      'Sophisticated': 4,
      'Agressive': 4,
      'Fiery': 4,
      'Urgent': 3,
      'Rowdy': 4,
      'Sentimental': 4,
      'Easygoing': 1,
      'Melancholy': 4,
      'Romantic': 2,
      'Peaceful': 1,
      'Brooding': 4,
      'Upbeat': 5,
      'Stirring': 5,
      'Lively': 5,
      'Other': 0,
      '': 0
})
df['tempoEncoded'] = df['tempo'].map({
      'Fast Tempo': 0,
      'Medium Tempo': 2,
      'Slow Tempo': 1,
      '': 0
})
df['genreEncoded'] = df['genre'].map({
      'Urban': 4,
      'Pop': 3,
      'Traditional': 2,
      'Alternative & Punk': 1,
      'Electronica': 1,
      'Rock': 1,
      'Soundtrack': 0,
      'Jazz': 0,
      'Other': 0,
      '': 0
})
df['artist_typeEncoded'] = df['artist_type'].map({
      'Female': 2,
      'Male': 3,
      'Mixed': 1,
      '': 0
}).astype('int64')
df.loc[df['edad_en_billboard'] <= 21, 'edadEncoded'] = 0
df.loc[(df['edad_en_billboard'] > 21) & (df['edad_en_billboard'] <= 26), 'edadEncoded'] = 1
df.loc[(df['edad_en_billboard'] > 26) & (df['edad_en_billboard'] <= 30), 'edadEncoded'] = 2
df.loc[(df['edad_en_billboard'] > 30) & (df['edad_en_billboard'] <= 40), 'edadEncoded'] = 3
df.loc[df['edad_en_billboard'] > 40, 'edadEncoded'] = 4
df.loc[df['durationSeg'] <= 150, 'durationEncoded'] = 0
df.loc[(df['durationSeg'] > 150) & (df['durationSeg'] <= 180), 'durationEncoded'] = 1
df.loc[(df['durationSeg'] > 180) & (df['durationSeg'] <= 210), 'durationEncoded'] = 2
df.loc[(df['durationSeg'] > 210) & (df['durationSeg'] <= 240), 'durationEncoded'] = 3
df.loc[(df['durationSeg'] > 240) & (df['durationSeg'] <= 270), 'durationEncoded'] = 4
df.loc[(df['durationSeg'] > 270) & (df['durationSeg'] <= 300), 'durationEncoded'] = 5
df.loc[df['durationSeg'] > 300, 'durationEncoded'] = 6
drop_elements = ['id',
                 'title',
                 'artist',
                 'mood',
                 'tempo',
                 'genre',
                 'artist_type',
                 'chart_date',
                 'anioNacimiento',
                 'durationSeg',
                 'edad_en_billboard']
df_encoded = df.drop(drop_elements, axis=1)
print(df[['moodEncoded', 'top']]
      .groupby(['moodEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))
print(df[['genreEncoded', 'top']]
      .groupby(['genreEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))
print(df[['tempoEncoded', 'top']]
      .groupby(['tempoEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))
print(df[['durationEncoded', 'top']]
      .groupby(['durationEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))
print(df[['edadEncoded', 'top']]
      .groupby(['edadEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))
print(df[['artist_typeEncoded', 'top']]
      .groupby(['artist_typeEncoded'], as_index=False)
      .agg(['mean', 'count', 'sum']))

# Creating the tree
cv = KFold(n_splits=10)
acc = list()
max_attributes = len(list(df_encoded))
depth_range = range(1, max_attributes + 1)
for depth in depth_range:
      fold_accuracy = []
      tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                               min_samples_split=20,
                                               min_samples_leaf=5,
                                               max_depth=depth,
                                               class_weight={1:3.5})
      for train_fold, valid_fold in cv.split(df_encoded):
            f_train = df_encoded.loc[train_fold]
            f_valid = df_encoded.loc[valid_fold]
            model = tree_model.fit(X=f_train.drop(['top'], axis=1),
                                   y=f_train['top'])
            valid_acc = model.score(X=f_valid.drop(['top'], axis=1),
                                    y=f_valid['top'])
            fold_accuracy.append(valid_acc)
      avg = sum(fold_accuracy) / len(fold_accuracy)
      acc.append(avg)
df2 = pd.DataFrame({'Max Depth': depth_range, 'Average Accuracy': acc})
df2 = df2[['Max Depth', 'Average Accuracy']]
print(df2.to_string(index=False))

# Visualizing the tree
y_train = df_encoded['top']
x_train = df_encoded.drop(['top'], axis=1).values
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                            min_samples_split=20,
                                            min_samples_leaf=5,
                                            max_depth=4,
                                            class_weight={1:3.5})
decision_tree.fit(x_train, y_train)
with open(r'tree1.dot', 'w') as f:
      f = tree.export_graphviz(decision_tree,
                               out_file=f,
                               max_depth=7,
                               impurity=True,
                               feature_names=list(df_encoded.drop(['top'], axis=1)),
                               class_names=['No', 'N1 Billboard'],
                               rounded=True,
                               filled=True)
render('dot', 'png', 'tree1.dot')

# Analysis
print(f'Decision tree accuracy: {round(decision_tree.score(x_train, y_train)*100, 2)}')

# Predictions
x_test = pd.DataFrame(columns=('top', 'moodEncoded', 'tempoEncoded', 'genreEncoded', 'artist_typeEncoded', 'edadEncoded', 'durationEncoded'))
x_test.loc[0] = (1, 5, 2, 4, 1, 0, 3)
y_pred = decision_tree.predict(x_test.drop(['top'], axis=1))
y_prob = decision_tree.predict_proba(x_test.drop(['top'], axis=1))
print(f'Prediction: {str(y_pred)}',
      f'Probability: {y_prob[0][y_pred]*100}%')
x_test2 = pd.DataFrame(columns=('top', 'moodEncoded', 'tempoEncoded', 'genreEncoded', 'artist_typeEncoded', 'edadEncoded', 'durationEncoded'))
x_test2.loc[0] = (0, 4, 2, 1, 3, 2, 3)
y_pred2 = decision_tree.predict(x_test2.drop(['top'], axis=1))
y_prob2 = decision_tree.predict_proba(x_test2.drop(['top'], axis=1))
print(f'Prediction: {str(y_pred2)}',
      f'Probability: {y_prob2[0][y_pred]*100}%')


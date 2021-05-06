import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %% Adat beolvasás, tisztítás

data = pd.read_csv('tracks.csv', encoding='utf-8')
missing = data.isnull().sum()
# print(missing) # -> csak a name oszlopban vannak Null értékek
# data['name'].replace('', np.nan, inplace=True)
data.dropna(subset=['name'], inplace=True)
data.drop(['id_artists', 'key', 'mode', 'time_signature'], 'columns', inplace=True)
#data['explicit'] = data['explicit'].astype('bool')
data['release_date'] = data['release_date'].str[:4]
# %% Bar chart
df2 = data
df2['decade'] = (df2['release_date'].str[:3] + '0').astype('int')
data.drop(data[data['decade'] == 1900].index, inplace=True)
decade_list = df2['decade'].drop_duplicates().tolist()
decade_list.sort()
print(df2['decade'].value_counts().sort_values())
count = df2['decade'].value_counts().sort_index()
plt.gcf().set_size_inches(14, 10)

count.plot.bar(color='lightseagreen', zorder=10)
plt.grid(zorder=0, color='gainsboro', linestyle='--')
plt.title('Zeneszámok megoszlása évtizedenként', fontsize=24, pad=13)
plt.ylabel('Darabszám', fontsize=22, labelpad=10)
plt.xlabel('Évtizedek', fontsize=22, labelpad=10)
plt.yticks(rotation=45, fontsize=16)
plt.xticks(rotation=45, fontsize=16)

plt.figure(figsize=(14, 7))
plt.show()

# %% Boxplot

twenty = df2[df2['decade'] >= 2000]

plt.figure(figsize=(12, 10))
twenty.drop(['duration_ms', 'loudness', 'decade', 'popularity', 'tempo', 'explicit', 'instrumentalness'], 'columns',
            inplace=True)
bp = twenty.boxplot(notch=True, vert=True, patch_artist=True,
                    medianprops=dict(linestyle='solid', linewidth=4, color='y'),
                    whiskerprops=dict(linestyle='solid', linewidth=1, color='k'),
                    capprops=dict(linestyle='solid', linewidth=4, color='orange'),
                    boxprops=dict(color='black'),
                    flierprops=dict(markerfacecolor='r', marker='s', mew=0.5))
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)

plt.show()

# %% Trendek

df = data.copy()

df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df.apply(lambda row: row.release_date.year, axis=1)
year_avg = df[["acousticness", "danceability", "energy", "instrumentalness", "liveness", "tempo", "valence",
               "loudness", "speechiness", "year"]].groupby("year").mean().sort_values(by="year").reset_index()

plt.figure(figsize=(14, 8))
plt.title("Zenei trendek az éveken át", fontsize=22)

lines = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "valence", "speechiness"]

for line in lines:
    ax = sns.lineplot(x='year', y=line, data=year_avg)

plt.xlabel("Évszámok", labelpad=12, fontsize=16)
plt.ylabel("Értékek 0 és 1 között", labelpad=12, fontsize=16)
plt.legend(lines)
plt.axis([1930, 2020, 0, 1])
plt.show()

# %% Heatmap
twenty = df2[df2['decade'] >= 2000]
corr = twenty.corr()
plt.figure(figsize=(12, 11))
sns.color_palette("icefire", as_cmap=True)
sns.heatmap(corr, vmax=1, vmin=-1, center=0, linewidth=.5, square=True, annot=True, annot_kws={'size': 8}, fmt='.1f',
            cmap='icefire')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=45, fontsize=12)
plt.title('Korrelációs mátrix', fontsize=25, pad=20)
plt.show()

# %% Train and test

model_df = twenty[['explicit', 'danceability', 'speechiness']]
X = model_df.drop('explicit', 'columns')
y = model_df['explicit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# %% Logistic Regression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
conf_y_test = y_test.to_numpy(copy=True)
confusion = confusion_matrix(conf_y_test, y_pred)
print(confusion)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion, cmap='BrBG_r')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion[i, j], ha='center', va='center', color='white')
plt.title('Confusion Matrix of Predicted Explicity with Logistic Regression')
plt.show()

# %% Lasso Regularization

lasso = LogisticRegression(penalty='l1', solver='liblinear')
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
conf_y_test = y_test.to_numpy(copy=True)
confusion = confusion_matrix(conf_y_test, y_pred)
print(confusion)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion, cmap='BrBG_r')
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion[i, j], ha='center', va='center', color='white', fontsize=13)
plt.title('Confusion Matrix of Logistic Regression', fontsize=16)
plt.show()


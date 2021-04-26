# %% Importok
import numpy as np
import pandas as pd

# %% Adat beolvasás, tisztítás
data = pd.read_csv('tracks.csv', encoding='utf-8')
missing = data.isnull().sum()
# print(missing)
data['name'].replace('', np.nan, inplace=True)
data.dropna(subset=['name'], inplace=True)
data.drop(['id_artists', 'key', 'mode', 'time_signature'], 'columns', inplace=True)


#%%
from pathlib import Path
import pandas as pd

#%%
data = pd.read_csv(
    Path('./data/train.csv')
)

#%%
data.info()
# O conjunto de dados contém 76_020 linhas e 371 colunas, sendo que todas as colunas são do tipo float64 ou int64.

#%%
print(data['ID'].describe())
data['ID'].hist()
# É interessante notar que os IDs não coincidem com os índices da tabela, porém estão uniformemente distribuídos. Isso provavelmente quer dizer que não são relevantes.

#%%
print(data['TARGET'].value_counts())
# Podemos ver que o conjunto é altamente desbalanceado, tendo apenas cerca de 3008 / (73_012 + 3008) ~ 4% de clientes satisfeitos.

#%%

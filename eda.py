#%%
import re
import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import curdoc
from bokeh.themes import Theme
from sklearn.feature_selection import VarianceThreshold
import eda_tools

output_notebook()
curdoc().theme = Theme("bokeh_theme.yaml")

#%%
data = pd.read_csv("./data/train.csv")
data.columns = data.columns.str.lower()

#%% [markdown]
"""
# Exploração inicial.
"""
#%% [markdown]
"""
A classificação dos clientes é dada pela variável "target". O valor "1" representa a classe positiva, que neste caso engloba os clientes _insatisfeitos_. O valor "0" então representa a classe negativa, dos clientes _satisfeitos_.
"""

#%%
X = data.drop(columns=["target"])
y = data["target"]

#%% [markdown]
"""
O conjunto de dados contém 76_020 linhas (clientes) e 370 colunas (variáveis), sendo que todas as colunas são do tipo float64 (111) ou int64 (259).
"""
#%%
X.info()

#%% [markdown]
"""
Os valores de "id" são todos distintos (como era de se esperar) e não coincidem com o índice da tabela.
"""
#%%
print(f"Número de 'id' distintos: {X['id'].nunique()}")

coinciding_indices = sum(X["id"] == X.index)
print(f"Número de 'id' que coincidem com o índice do data frame: {coinciding_indices}")

#%% [markdown]
"""
Os índices estão uniformemente distribuídos. Isso provavelmente quer dizer que não são relevantes e podem ser descartados.
"""
#%%
X["id"].plot.hist()
X = X.drop(columns=["id"])

#%% [markdown]
"""
Podemos ver que todas as variáveis deste conjunto contêm o termo "var" seguido de um número.
"""
#%%
variables = []
no_var = 0
for feature in X.columns:
    result = re.search(r"var(\d+)", feature)
    if result is not None:
        var_number = int(result.group(1))
        variables.append(var_number)
    else:
        print(f"{feature} não contém 'var'.\n")
        no_var += 1
else:
    variables = sorted(set(variables))

print(
    f"Variáveis presentes: {variables}\n",
    f"Número de variáveis básicas no conjunto: {len(variables)}\n",
    f"Número de variáveis que não são função de 'var#': {no_var}",
    sep="\n",
)

#%% [markdown]
"""
Também podemos ver a única variável que falta no intervalo que temos é a "var23".
"""
#%%
for idx in range(1, variables[-1] + 1):
    if idx not in variables:
        print(f"'var{idx}' não é uma variável neste conjunto.")

#%% [markdown]
"""
Podemos ver que o conjunto é altamente desbalanceado, tendo apenas cerca de 3008 / 76_020 ~ 4% de clientes insatisfeitos. Com esse nível de desbalanceamento, não será adequado utilizar a acurácia como métrica.
"""
#%%
print(
    "Número de clientes em cada classe:",
    y.value_counts(),
    sep="\n",
)

#%% [markdown]
"""
As variáveis desta tabela não tem nomes claros devido à anonimização dos dados. Isso faz com que qualquer intuição a respeito delas seja muito difícil.
"""
#%%
print(data.columns)

#%% [markdown]
"""
Para ter alguma intuição sobre o conjunto de dados, nós podemos verificar a existência de variáveis que apresentem valores constantes para todos os clientes. Essas variáveis podem ser facilmente descartadas, já que não tem poder preditivo. As transformações do scikit-learn não retornam DataFrames, então é preciso de um pouco de trabalho para manter os nomes das colunas.
"""
#%%
cols = X.columns
original_len = len(cols)
selector = VarianceThreshold()
selector.fit(X)
X = pd.DataFrame(selector.transform(X))
X.columns = cols[selector.get_support()]
new_len = len(X.columns)
num_elim_var = original_len - new_len
print(f"Número de variáveis eliminadas: {num_elim_var}")
print(f"Número de variáveis restantes: {new_len}")

#%% [markdown]
"""
Como ainda resta um número muito grande de variáveis (335) e elas são de difícil interpretação, podemos buscar quais delas são individualmente relevantes para a tarefa de classificação. Isso pode ser feito facilmente com uma árvore de decisão, utilizando as variáveis uma a uma.

É importante notar que, como as classes são desbalanceadas, o modelo não vai conseguir se ajustar bem naturalmente. Por isso devemos especificar que as classes devem receber pesos de maneira inversamente proporcional à sua predominância.

Vamos também utilizar árvores com apenas um nó, para tentar identificar variáveis que façam um divisão binária simples nos dados.

Como já comentado, o fato de o conjunto de dados ser tão desbalanceado jaz com que a acurácia não seja uma boa métrica. Por esse motivo, vamos analisar três outras figuras de mérito em conjunto: a sensibilidade, a especificidade e a área sob a curva ROC, que será usada para ordenar o DataFrame.

A última coluna do DataFrame abaixo apresenta o valor da variável que divide o conjunto de dados nas partes de classe negativa e positiva.
"""

#%%
df, trees = eda_tools.create_trees(X, y, features=X.columns, random_state=101)

df

#%% [markdown]
"""
Analisando esses resultados vemos que existem algumas variáveis que aparentam ter uma capacidade de classificação razoável por si só.

Por exemplo, vamos considerar a variável "saldo\_var30", que é a que possui a maior área sob a curva ROC. O gráfico mostra o percentual acumulado de clientes que tem um saldo de _até_ um certo valor (representado no eixo x). Podemos ver que a proporção de clientes nos dois grupos apresenta um salto grande de prevalência em torno do valor 0, mas o grupo insatisfeito aumenta muito mais acentuadamente. Isso significa que temos uma concentração proporcionalmente maior de clientes insatisfeitos. Em outras palavras, clientes insatisfeitos tendem a ter um _saldo menor_.
"""
#%%
feature = df.iloc[0]
eda_tools.plot_feature_cumsum(X, y, feature["feature"], feature["cutoff"])

#%% [markdown]
"""
Também é interessante observar a coluna 'var15'. A sua distribuição de valores sugere que ela possa ser a idade do cliente. A árvore de decisão que treinamos encontrou uma fronteira de decisão no valor de 26.5, sendo que clientes abaixo dessa idade seriam classificados como satisfeitos.
"""
#%%
feature = df.loc[1, :]
eda_tools.plot_feature_cumsum(
    X, y, feature["feature"], feature["cutoff"], x_range=(5, 60)
)

#%% [markdown]
"""
Dependendo do objetivo do projeto, uma simples árvore de decisão dessas já poderia ser suficiente. Como o nosso objetivo é alcançar um desempenho de 70%, precisamos buscar um modelo um pouco mais sofisticado.
"""

#%%

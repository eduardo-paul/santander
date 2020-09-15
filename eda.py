#%%
import re
from pathlib import Path
import numpy as np
import pandas as pd
from bokeh.io import output_notebook
from bokeh.models import Span
from bokeh.plotting import curdoc, figure, show
from bokeh.themes import Theme
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import eda_tools

output_notebook()
curdoc().theme = Theme(Path('bokeh_theme.yaml'))

#%%
data = pd.read_csv(
    Path('./data/train.csv')
)

#%%
# A classificação dos clientes é dada pela variável TARGET.
X = data.drop(columns=['TARGET'])
y = data['TARGET']

#%%
X.info()
# O conjunto de dados contém 76_020 linhas e 370 colunas, sendo que todas as colunas são do tipo float64 (111) ou int64 (259).

#%%
# Os IDs são todos distintos (como era de se esperar) e não coincidem com o índice da tabela.
print(f"Número de IDs distintos: {X['ID'].nunique()}")

coinciding_indices = sum(X['ID'] == X.index)
print(f'Número de IDs que coincidem com o índice do data frame: {coinciding_indices}')

#%%
# Os índices estão uniformemente distribuídos. Isso provavelmente quer dizer que não são relevantes e podem ser descartados.
X['ID'].hist()
X = X.drop(columns=['ID'])

#%%
# Podemos ver que todas as variáveis deste conjunto, com a exceção de "ID" e "TARGET," contêm a palavra "var" seguida de um número.
vars = re.compile(r'var(\d+)')
variables = []
num = 0
for feature in data.columns:
    result = re.search(vars, feature)
    if result is not None:
        variables.append(
            int(result.group(1))
        )
        num += 1
    else:
        print(f'{feature} does not contain a "var."\n')
else:
    variables = sorted(set(variables))

print(
    f'Variables present: {variables}\n',
    f'Number of features containing "var": {num}',
    sep='\n',
)

#%%
# Também podemos ver a única variável que falta no intervalo que temos é a var23.
for idx in range(1, variables[-1]+1):
    if idx not in variables:
        print(f'{idx} is not a variable in the dataset.')

#%%

#%%
# Podemos ver que o conjunto é altamente desbalanceado, tendo apenas cerca de 3008 / 76_020 ~ 4% de clientes insatisfeitos. Com esse nível de desbalanceamento, não será adequado utilizar a acurácia como métrica.
print(
    f"Número de clientes em cada classe:",
    y.value_counts(),
    sep='\n',
)

#%%
# As variáveis desta tabela não tem nomes claros devido à anonimização dos dados. Isso faz com que qualquer intuição a respeito delas seja muito difícil.
print(data.columns)

#%%
# Para ter alguma intuição sobre o conjunto de dados, nós podemos verificar a existência de variáveis que apresentem valores constantes para todos os clientes. Essas variáveis podem ser facilmente descartadas, já que não tem poder preditivo.
const_var = VarianceThreshold()
data01 = const_var.fit_transform(data)
num_elim_var = data.shape[1] - data01.shape[1]
print(f'Número de variáveis eliminadas: {num_elim_var}')
print(f'Número de variáveis restantes: {data01.shape[1]}')

#%%
# Como ainda resta um número muito grande de variáveis (337) e elas são de difícil interpretação, podemos buscar quais delas são individualmente relevantes para a tarefa de classificação. Isso pode ser feito facilmente com uma árvore de decisão, utilizando as variáveis uma a uma.
# É importante notar que como as classes são desbalanceadas, o modelo não vai conseguir se ajustar bem naturalmente. Por isso devemos especificar que as classes devem receber pesos de maneira inversamente proporcional à sua predominância.
# Vamos também utilizar árvores com apenas um nó, para tentar identificar variáveis que façam um divisão binária simples nos dados.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

tree = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=1,
    random_state=102,
)

results = pd.DataFrame(columns=['feature', 'sensitivity', 'specificity', 'roc_auc'])
for feature in X_train.columns:
    # Treino.
    tree.fit(X_train[[feature]], y_train)
    # Predição.
    y_pred = tree.predict(X_test[[feature]])
    # Dicionário com as métricas.
    report = eda_tools.report(y_test, y_pred)
    # Acrescentando o nome da variável ao dicionário.
    report['feature'] = feature

    new_row = pd.DataFrame(report)
    results = results.append(new_row, ignore_index=True)
else:
    results = results.sort_values(by='roc_auc', ascending=False)

print(results)

#%%
# Analisando as árvores de decisão vemos que existem algumas variáveis que aparentam ter uma capacidade de classificação razoável por si só. Vamos analisar as que apresentam a maior área sob a curva ROC.

num_of_features = 10

best_features = pd.DataFrame(columns=['feature', 'sensitivity', 'specificity', 'roc_auc', 'cutoff'])

trees = {
    feature: DecisionTreeClassifier(
        class_weight='balanced',
        max_depth=1,
        random_state=102,
    )
    for feature in results['feature'][:num_of_features]
}

for feature in trees:
    # Treino.
    trees[feature].fit(X_train[[feature]], y_train)
    # Predição.
    y_pred = trees[feature].predict(X_test[[feature]])
    # Dicionário com as métricas.
    report = eda_tools.report(y_test, y_pred)
    # Acrescentando o nome da variável ao dicionário.
    report['feature'] = feature
    report['cutoff'] = eda_tools.tree_cutoff(trees[feature])

    new_row = pd.DataFrame(report)
    best_features = best_features.append(new_row, ignore_index=True)
else:
    best_features = best_features.sort_values(by='roc_auc', ascending=False)

print(best_features)

#%%
# Vamos analisar as primeiras variáveis para buscar alguma intuição sobre elas.
for idx in range(10):
    feature = best_features.iloc[idx]
    eda_tools.plot_feature_cumsum(data, feature['feature'], feature['cutoff']) 

#%%
# Dentre as primeiras colocadas nesta lista de variáveis, o saldo_var5 apresenta a maior sensibilidade, sem perder muito em termos de especificidade.

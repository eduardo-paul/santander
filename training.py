#%% [markdown]
"""
# Treinamento.
Como vimos na análise exploratória, uma árvore de decisão que considere apenas uma variável e tenha uma profundidade (parâmetro "max_depth") de dois níveis já é capaz de alcançar resultados razoáveis. Neste documento vamos treinar árvores de decisão com várias profundidades e que se utilizem de todas as variáveis disponíveis no conjunto de dados, com exceção das que apresentam variância nula. A expectativa é encontrar uma árvore de decisão simples que otimize a sensibilidade das previsões.
"""

#%%
from pathlib import Path
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import eda_tools

data = pd.read_csv(Path("./data/train.csv"))
data.columns = data.columns.str.lower()

X = data.drop(columns=["id", "target"])
y = data["target"]

pipeline = Pipeline(
    [
        ("var", VarianceThreshold()),
        ("tree", DecisionTreeClassifier(class_weight="balanced")),
    ]
)

param_grid = {
    "tree__max_depth": range(2, 10),
}

scoring = {
    "sensitivity": make_scorer(recall_score),
    "specificity": make_scorer(eda_tools.specificity_score),
    "roc_auc": make_scorer(roc_auc_score),
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=scoring,
    n_jobs=-1,
    refit="roc_auc",
)

grid_search.fit(X, y)

#%% [markdown]
"""
Os resultados mostram que com uma simples árvore de decisão já somos capazes de alcançar uma área sob a curva ROC de 0,76, com sensibilidade e especificidade comparáveis.
"""

#%%
results = pd.DataFrame(grid_search.cv_results_).sort_values(by="rank_test_roc_auc")

print("Melhor árvore encontrada:")
results[
    ["params", "mean_test_sensitivity", "mean_test_specificity", "mean_test_roc_auc"]
].iloc[0, :]

#%% [markdown]
"""
Analisando a importância das variáveis segundo a árvore de decisão, vemos que de fato as variáveis "var15" (que associamos à idade) e "saldo_var30" são as mais relevantes para a classificação dos clientes.
"""

#%%
tree = grid_search.best_estimator_.named_steps["tree"]
selector = grid_search.best_estimator_.named_steps["var"]
X_cols = X.columns[selector.get_support()]

feature_importances = pd.DataFrame(
    {"importance": tree.feature_importances_, "feature": X_cols}
).sort_values(by="importance", ascending=False, ignore_index=True)

feature_importances = feature_importances[feature_importances["importance"] != 0]

feature_importances

#%% [markdown]
"""
Enviando o arquivo com as previsões para o Kaggle, a área sob a curva ROC no conjunto de teste foi de 0,74747. Esse resultado está dentro da meta estabelecidade no início deste projeto, de forma que ele pode ser considerado um sucesso.
"""

#%%
test_data = pd.read_csv(Path("./data/test.csv"))

X_test = test_data.drop(columns=["ID"])

predictions = grid_search.best_estimator_.predict(X_test)

submission = pd.concat([test_data["ID"], pd.Series(predictions, name="TARGET")], axis=1)

submission.to_csv(
    Path("submission.csv"),
    index=False,
)

from pandas import concat
from pandas import DataFrame
from bokeh.models import Span
from bokeh.models.annotations import Label
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


def report(y_true, y_pred):
    """Report relevant training results: sensitivity, specificity and area under ROC curve."""
    conf_matrix = confusion_matrix(y_true, y_pred)

    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fn = conf_matrix[1][0]
    fp = conf_matrix[0][1]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    report = {
        "sensitivity": [sensitivity],
        "specificity": [specificity],
        "roc_auc": [roc_auc_score(y_true, y_pred)],
    }

    return report


def create_trees(X, y, features, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    trees = {
        feature: DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=1,
            random_state=random_state,
        )
        for feature in features
    }

    df = DataFrame(
        columns=["feature", "sensitivity", "specificity", "roc_auc", "cutoff"]
    )

    for feature in trees:
        trees[feature].fit(X_train[[feature]], y_train)
        y_pred = trees[feature].predict(X_test[[feature]])
        rep = report(y_test, y_pred)
        rep["feature"] = feature
        rep["cutoff"] = tree_cutoff(trees[feature])
        new_row = DataFrame(rep)
        df = df.append(new_row, ignore_index=True)
    else:
        df = df.sort_values(by="roc_auc", ascending=False)

    return df, trees


def specificity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]

    specificity = tn / (tn + fp)

    return specificity


def tree_cutoff(tree):
    tree_text = export_text(tree)
    rows = tree_text.split("\n")
    cutoff = rows[0].split()[-1]
    return float(cutoff)


def plot_feature(data, best_features, feature, xmin, xmax):
    _, ax = plt.subplots()
    for target, ls in zip([0, 1], ["b", "r"]):
        dataset = data[data["target"] == target][feature]

        sns.histplot(
            dataset,
            binwidth=1,
            stat="probability",
            alpha=0.5,
            color=ls,
            label=target,
            ax=ax,
        )

    ax.axvline(
        x=best_features[best_features["feature"] == feature].iloc[0, -1],
        alpha=0.3,
        color="k",
        lw=1.5,
        ls=":",
    )

    ax.set_xlim(xmin, xmax)
    ax.legend()


# def get_feature_cumsum(data, feature):
def get_feature_cumsum(X, y, feature):
    """Return a pandas.DataFrame containing columns for the cumulative sum for each target class."""
    data = concat([X, y], axis=1)
    feature_data = data[[feature, "target"]].sort_values(by=feature)
    feature_data["0"] = -(feature_data["target"] - 1).cumsum()
    feature_data["0"] = feature_data["0"] / feature_data["0"].iloc[-1]
    feature_data["1"] = feature_data["target"].cumsum()
    feature_data["1"] = feature_data["1"] / feature_data["1"].iloc[-1]
    feature_data = feature_data.drop(columns=["target"])
    feature_data = feature_data.drop_duplicates(
        subset=feature, keep="last"
    ).reset_index(drop=True)
    return feature_data


def plot_feature_cumsum(X, y, feature, cutoff):
    feature_data = get_feature_cumsum(X, y, feature)

    p = figure(
        x_range=(-1, 4),
    )

    p.line(
        feature_data[feature],
        feature_data["0"],
        legend_label="0",
        line_color="navy",
    )

    p.line(
        feature_data[feature],
        feature_data["1"],
        legend_label="1",
        line_color="crimson",
    )

    span = Span(
        location=cutoff,
        dimension="height",
        line_color="gray",
        line_dash="dotted",
    )

    label = Label(
        text="limiar da árvore\nde decisão",
        text_align='right',
        x=cutoff - 0.1,
        y=0.9,
        x_units="data",
    )

    p.add_layout(span)
    p.add_layout(label)

    p.xaxis.axis_label = feature
    p.yaxis.axis_label = "soma cumulativa"

    p.legend.location = "top_left"

    show(p)

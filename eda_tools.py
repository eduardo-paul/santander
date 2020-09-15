from bokeh.models import Span
from bokeh.models.annotations import Label
from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

def report(y_true, y_pred):
    '''Report relevant training results: sensitivity, specificity and area under ROC curve.'''
    conf_matrix = confusion_matrix(y_true, y_pred)

    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fn = conf_matrix[1][0]
    fp = conf_matrix[0][1]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    report = {
        'sensitivity': [sensitivity],
        'specificity': [specificity],
        'roc_auc': [roc_auc_score(y_true, y_pred)],
    }

    return report

def specificity_score(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)

    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]

    specificity = tn / (tn + fp)

    return specificity

def tree_cutoff(tree):
    tree_text = export_text(tree)
    rows = tree_text.split('\n')
    cutoff = rows[0].split()[-1]
    return float(cutoff)

def plot_feature(data, best_features, feature, xmin, xmax):
    fig, ax = plt.subplots()
    for target, ls in zip([0, 1], ['b', 'r']):
        dataset = data[data['TARGET'] == target][feature]

        sns.histplot(
            dataset,
            binwidth=1,
            stat='probability',
            alpha=.5,
            color=ls,
            label=target,
            ax=ax,
        )

    ax.axvline(
        x=best_features[best_features['feature'] == feature].iloc[0, -1],
        alpha=.3,
        color='k',
        lw=1.5,
        ls=':'
    )

    ax.set_xlim(xmin, xmax)
    ax.legend()

def get_feature_cumsum(data, feature):
    '''Return a pandas.DataFrame containing columns for the cumulative sum for each target class.'''
    feature_data = data[[feature, 'TARGET']].sort_values(by=feature)
    feature_data['0'] = -(feature_data['TARGET']-1).cumsum()
    feature_data['0'] = feature_data['0'] / feature_data['0'].iloc[-1]
    feature_data['1'] = feature_data['TARGET'].cumsum()
    feature_data['1'] = feature_data['1'] / feature_data['1'].iloc[-1]
    feature_data = feature_data.drop(columns=['TARGET'])
    feature_data = feature_data.drop_duplicates(subset=feature, keep='last').reset_index(drop=True)
    return feature_data

def plot_feature_cumsum(data, feature, cutoff):
    feature_data = get_feature_cumsum(data, feature)

    p = figure(
        x_range=(-1, 4),
    )

    p.line(
        feature_data[feature],
        feature_data['0'],
        legend_label='0',
        line_color='navy',
    )

    p.line(
        feature_data[feature],
        feature_data['1'],
        legend_label='1',
        line_color='crimson',
    )

    span = Span(
        location=cutoff,
        dimension='height',
        line_color='gray',
        line_dash='dotted',
    )

    label = Label(
        text='Decision Tree cutoff',
        x=cutoff + .1,
        y=0.9,
        x_units='data',
    )

    p.add_layout(span)
    p.add_layout(label)

    p.xaxis.axis_label = feature
    p.yaxis.axis_label = 'cumulative number'

    p.legend.location = 'top_left'

    show(p)

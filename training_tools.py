from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

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

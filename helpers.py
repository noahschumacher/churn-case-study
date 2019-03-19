import numpy as np

## Creating confusion matrix from predicts and trues
def standard_confusion_matrix(y_true, y_predict):
    y_true, y_predict = y_true>0, y_predict>0

    tp, fp = (y_true & y_predict).sum(), (~y_true & y_predict).sum()
    fn, tn = (y_true & ~y_predict).sum(), (~y_true & ~y_predict).sum()
    return np.array([[tp, fp], [fn, tn]])


## Returns dictionary of threshold and assocated profit val for given cost_ben
def profit_curve(cost_ben, pred_probs, labels):
    order = pred_probs.argsort()[::0-1]	## list of indexes to sort in reverse order
    thresholds, profits = [], []

    #print(pred_probs)
    #pred_probs = pred_probs[:, 1]
    for ind in order:
        thresh = pred_probs[ind]	## Sets theshold to prob
        pos_class = pred_probs > thresh
        confusion_mat = standard_confusion_matrix(labels, pos_class)
        profit = (confusion_mat * cost_ben).reshape(1,-1).sum()/len(labels)
        
        profits.append(profit)
        thresholds.append(thresh)
    return (thresholds, profits)


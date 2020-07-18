import getpass

import pandas as pd
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.tree import export_graphviz
from sklearn.metrics import average_precision_score
# INTENTAR REMOVER MODULO
def print_roc_auc(labels, preds):
    fpr, tpr, threshold = roc_curve(labels, preds)
    assert(auc(fpr, tpr) == roc_auc_score(labels, preds))
    print('AUC --> {0}'.format(auc(fpr, tpr)))


def print_metrics(model, hfo_type_name, y_test, y_pred, y_probs):
    print('')
    print('-------------------------------------------')
    print('Displaying metrics for {0} using {1} model:'.format(hfo_type_name, model))
    #print('ROC AUC of ---> {0}'.format(roc_auc_score(y_test, y_probs)))
    print('Accuracy: {0}'.format(accuracy_score(y_test, y_pred)))
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
   # average_precision = average_precision_score(y_test, y_probs)
    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('-------------------------------------------')
    print('')


def print_feature_importances(model, feature_names):
    # Extract feature importances
    fi = pd.DataFrame({'feature': feature_names,
                       'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)

    print(fi)


def generate_trees(feature_list, train_features, train_labels,
                   amount=1, directory='/home/{user}'.format(user=getpass.getuser())):
    # Limit depth of tree to 3 levels
    rf_small = RandomForestClassifier(n_estimators=amount, max_depth=4)
    rf_small.fit(train_features, train_labels)
    for i in range(amount):
        # Extract the small tree
        tree_small = rf_small.estimators_[i]
        # Save the tree as a png image
        out_path = '{dir}/thesis_tree_{k}.dot'.format(dir=directory, k=i)
        export_graphviz(tree_small,
                        out_file=out_path,
                        feature_names=feature_list,
                        rounded=True,
                        precision=1)
        (graph,) = pydot.graph_from_dot_file(out_path)
        graph.write_png('{dir}/thesis_tree_{k}.png'.format(dir=directory, k=i))

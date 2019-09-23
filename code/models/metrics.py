import getpass
import pandas as pd
import pydot
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score, roc_curve, auc

def print_auc_0(labels, rates):
    fpr, tpr, threshold = roc_curve(labels, rates)
    print( 'AUC --> {0}'.format(auc(fpr, tpr)))


def print_metrics(y_test, y_pred, y_probs, hfo_type_name, model):
    print('')
    print('-------------------------------------------')
    print('Displaying metrics for {0} using {1} model:'.format(hfo_type_name, model))
    print('ROC AUC of ---> {0}'.format(roc_auc_score(y_test, y_probs)))
    print('Accuracy: {0}'.format( accuracy_score(y_test, y_pred)))
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('-------------------------------------------')
    print('')

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



def print_feature_importances(model, feature_namblock_durationes):
    # Extract feature importances
    fi = pd.DataFrame({'feature': feature_names,
                       'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)

    # Display
    fi.head()

    '''    
    #Get importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    '''
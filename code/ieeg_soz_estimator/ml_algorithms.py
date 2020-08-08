import getpass

import pydot
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import export_graphviz
from xgboost.sklearn import XGBClassifier
from random import random, choices
import pandas as pd


# Machine learning sklearn algorithms

def naive_bayes(train_features, train_labels, test_features, feature_list=None,
                hfo_type_name=None):
    clf = GaussianNB()
    clf.fit(train_features, train_labels)
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]
    return clf_predictions, clf_probs, clf


def svm_m(train_features, train_labels, test_features, feature_list=None,
          hfo_type_name=None):
    # kernel = [linear, poly, rbf, sigmoid]
    kernel = 'linear'
    # clf = svm.SVC(kernel=kernel, C=1, probability=True, degree= 3, gamma='auto')

    clf = LinearSVC(C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(clf)
    clf.fit(train_features, train_labels)

    clf_predictions = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        clf_probs = clf.predict_proba(test_features)[:, 1]
    else:
        clf_probs = None

    return clf_predictions, clf_probs, clf


def random_forest(train_features, train_labels, test_features,
                  feature_list=None, hfo_type_name=None):
    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion='gini',  # 'entropy'
        random_state=32,
        # max_features=None,
        bootstrap=True,  # Sampling with replacement for each tree
        n_jobs=-1,  # use all available processors
        # min_samples_split= 0.005,
        # min_samples_leaf= 0.005,
        verbose=0,
        oob_score=False,
        # Whether to use out-of-bag samples to estimate the generalization accuracy.
        # class_weight='balanced_subsample'
    )
    rf.fit(train_features, train_labels)

    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]

    # IF FEATURE IMPORTANCE FIGS NOT EXISTS
    #    print_feature_importances(rf, feature_list)
    #    graphics.feature_importances(feature_list, rf.feature_importances_, hfo_type_name)
    return rf_predictions, rf_probs, rf


def balanced_random_forest(train_features, train_labels, test_features,
                           feature_list=None, hfo_type_name=None):
    rf = BalancedRandomForestClassifier(
        random_state=32,
        n_jobs=-1,  # use all available processors
        # class_weight='balanced_subsample'
    )
    rf.fit(train_features, train_labels)
    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]
    # IF FEATURE IMPORTANCE FIGS NOT EXISTS
    # print_feature_importances(rf, feature_list)
    # graphics.feature_importances(feature_list, rf.feature_importances_, hfo_type_name)
    return rf_predictions, rf_probs, rf


def xgboost(train_features, train_labels, test_features, feature_list=None,
            hfo_type_name=None):
    clf = XGBClassifier(nthread=-1)
    '''
    #clf = XGBClassifier(learning_rate=0.05,
                        n_estimators=1000, #100
                        max_depth=6,
                        min_child_weight=3,
                        gamma=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.005,
                        objective='binary:logistic',
                        nthread=-1,
                        scale_pos_weight=1,
                        seed=10,
                        eval_metric='aucpr' #'aucpr'
                        )
    '''
    clf.fit(train_features, train_labels)
    # Predict over test
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    # graphics.feature_importances(feature_list, clf.feature_importances_, hfo_type_name, fig_id)
    return clf_predictions, clf_probs, clf


def sgd(train_features, train_labels, test_features, feature_list=None,
        hfo_type_name=None):
    clf = SGDClassifier(loss='modified_huber', max_iter=1000, n_jobs=-1)
    clf.fit(train_features, train_labels)
    # Predict over test
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    return clf_predictions, clf_probs, clf


# SIMULATOR

# Reviewed
def simulator(test_labels, distr, confidence):
    simulated_preds = []
    simulated_probs = []
    for soz_label in test_labels:
        r = random()
        if soz_label:
            if r <= confidence:
                pred = 1
                prob = choices(distr['TP'],
                               weights=[1 / len(distr['TP'])] * len(
                                   distr['TP']), k=1)[0]
            else:
                pred = 0
                prob = choices(distr['FN'],
                               weights=[1 / len(distr['FN'])] * len(
                                   distr['FN']), k=1)[0]
        else:
            if r <= confidence:
                pred = 0
                prob = choices(distr['TN'],
                               weights=[1 / len(distr['TN'])] * len(
                                   distr['TN']), k=1)[0]
            else:
                pred = 1
                prob = choices(distr['FP'],
                               weights=[1 / len(distr['FP'])] * len(
                                   distr['FP']), k=1)[0]

        assert (not isinstance(prob, list)) #choices returns a list of k values
        simulated_preds.append(pred)
        simulated_probs.append(prob)
    return simulated_preds, simulated_probs

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


################################################################################

# Default global var of the models of ml to run
models_to_run = ['XGBoost']  # 'Balanced random forest''Linear SVM'
models_dic = {'XGBoost': xgboost,
              'Linear SVM': svm_m,
              'Random Forest': random_forest,
              'Balanced random forest': balanced_random_forest,
              'SGD': sgd,
              'Bayes': naive_bayes,
              'Simulated': simulator
              }
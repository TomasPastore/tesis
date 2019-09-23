
from sklearn.ensemble import RandomForestClassifier
# from sklearn_pandas import DataFrameMapper, cross_val_score, crossvalpredict
from metrics import print_metrics, print_auc_0
from config import DEBUG
import graphics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from xgboost import XGBClassifier

from imblearn.ensemble import BalancedRandomForestClassifier


# Baseline
def hfo_rate(patients, hfo_type_name):
    labels = []
    hfo_rates = []
    for p in patients:
        if 'Interpolated' in p.id:
            continue
        for e in p.electrodes:
            labels.append(e.soz)
            hfo_rates.append(e.get_hfo_rate(hfo_type_name, p.file_blocks)) #Measured in events/min

    print('AUC for HFO rate model (Baseline):')
    print_auc_0(labels, hfo_rates)
    graphics.plot_roc(
        labels,
        hfo_rates,
        legend='{hfo_type_name} type, {n_electrodes} electrodes.'.format(hfo_type_name=hfo_type_name,
                                                                         n_electrodes=len(labels)),
        title='Baseline ROC curve for HFO rate in {hfo_type_name}.\n Hippocampus electrodes n={n_elec}.'.format(hfo_type_name=hfo_type_name, n_elec=len(labels))
    )

def naive_bayes(train_features, train_labels, test_features):
    clf = GaussianNB()
    clf.fit(train_features, train_labels)
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]
    return clf_predictions, clf_probs

def svm_m(train_features, train_labels, test_features):

    #kernel = [linear, poly, rbf, sigmoid]
    kernel = 'linear'
    clf = svm.SVC(kernel=kernel, C=1, probability=True, class_weight='balanced', degree= 3, gamma='auto')

    #clf = LinearSVC(C=1.0, class_weight='balanced')
    clf.fit(train_features, train_labels)

    clf_predictions = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        clf_probs = clf.predict_proba(test_features)[:,1]
    else:
        clf_probs= None

    return clf_predictions, clf_probs

def balanced_random_forest(train_features, train_labels, test_features):
    rf = BalancedRandomForestClassifier(
        n_estimators=1000,
        random_state=32,
        n_jobs=-1,  # use all available processors
        # class_weight='balanced_subsample'
    )
    rf.fit(train_features, train_labels)
    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]

    return rf_predictions, rf_probs

def xgboost(train_features, train_labels, test_features):
    clf = XGBClassifier()
    clf.fit(train_features, train_labels)
    # Predict over test
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    return clf_predictions, clf_probs

def random_forest(train_features, train_labels, test_features):

    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion='gini', #'entropy'
        random_state=32,
        #max_features=None,
        bootstrap=True, #Sampling with replacement for each tree
        n_jobs=-1, #use all available processors
        # min_samples_split= 0.005,
        # min_samples_leaf= 0.005,
        verbose=0,
        oob_score=False, #Whether to use out-of-bag samples to estimate the generalization accuracy.
        #class_weight='balanced_subsample'
    )

    rf.fit(train_features, train_labels)

    if DEBUG:
        print('Classes order: {0}'.format(rf.classes_))

    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]

    return rf_predictions, rf_probs

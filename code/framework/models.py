from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import  SGDClassifier
from xgboost import XGBClassifier

def naive_bayes(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):
    clf = GaussianNB()
    clf.fit(train_features, train_labels)
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]
    return clf_predictions, clf_probs

def svm_m(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):

    #kernel = [linear, poly, rbf, sigmoid]
    kernel = 'linear'
    #clf = svm.SVC(kernel=kernel, C=1, probability=True, degree= 3, gamma='auto')

    clf = LinearSVC(C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(clf)
    clf.fit(train_features, train_labels)

    clf_predictions = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        clf_probs = clf.predict_proba(test_features)[:,1]
    else:
        clf_probs= None

    return clf_predictions, clf_probs

printed = dict()
def random_forest(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):

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

    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]
    #if hfo_type_name not in printed.keys():
    #    printed[hfo_type_name] = True
    #    print_feature_importances(rf, feature_list)
    #    graphics.feature_importances(feature_list, rf.feature_importances_, hfo_type_name)
    return rf_predictions, rf_probs


def balanced_random_forest(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):
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

    #print_feature_importances(rf, feature_list)
    #graphics.feature_importances(feature_list, rf.feature_importances_, hfo_type_name)
    return rf_predictions, rf_probs

def xgboost(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):
    clf = XGBClassifier(nthread=-1)
    clf.fit(train_features, train_labels)
    # Predict over test
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    return clf_predictions, clf_probs

def sgd(train_features, train_labels, test_features, feature_list=None, hfo_type_name=None):
    clf = SGDClassifier(loss='modified_huber', max_iter=1000, n_jobs=-1)
    clf.fit(train_features, train_labels)
    # Predict over test
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    return clf_predictions, clf_probs
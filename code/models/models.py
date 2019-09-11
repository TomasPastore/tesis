import numpy as np
import pandas as pd
import math as mt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, cross_val_score
from metrics import print_metrics
from config import DEBUG


def segmentate(patients, train_p=0.6, test_p=0.2, val_p=0.2):
    assert (1 - train_p - test_p - val_p == 0)
    patient_count = len(patients)
    train_size = int(patient_count * train_p)
    test_size = int(patient_count * test_p)
    validation_size = test_size
    train_size += patient_count - (train_size + test_size + validation_size)

    train_set = patients[:train_size]
    test_set = patients[train_size:train_size + test_size]
    validation_set = patients[patient_count - validation_size:patient_count]
    return train_set, test_set, validation_set


# patients is a list
def segmentate_2(patients, train_p=0.6):
    assert (train_p <= 1 and train_p >= 0)
    patient_count = len(patients)
    train_size = int(patient_count * train_p)
    test_size = int(patient_count * (1 - train_p))
    train_size += patient_count - (train_size + test_size)
    train_set = patients[:train_size]
    test_set = patients[train_size:patient_count]
    return train_set, test_set

def get_features_and_labels(patients, hfo_type_name, feature_names):
    features = []
    labels = []
    for p in patients:
        for e in p.electrodes:
            for h in e.hfos[hfo_type_name]:
                feature_row_i = {}
                for feature_name in feature_names:
                    if 'angle' in feature_name or 'vs' in feature_name:
                        feature_row_i['SIN({0})'.format(feature_name)] = mt.sin(h.info[feature_name])
                        feature_row_i['COS({0})'.format(feature_name)] = mt.cos(h.info[feature_name])
                    else:
                        feature_row_i[feature_name] = h.info[feature_name]

                features.append(feature_row_i)
                labels.append({'soz': e.soz})
    return features, labels

'''
def get_clusters(patients, train=0.6):

features = []
for p in patients:
    for e in p.electrodes:
        for h in e.hfos['RonO']:
            features.append([h.info[feature] for feature in feature_names])
print(np.array(features).shape)

features = pd.DataFrame(features, columns=feature_names)
features.describe()
labels = np.array(features['soz'])
features = features.drop('soz', axis=1)
features = np.array(features)

train_features, test_features, \
train_labels, test_labels = train_test_split(features, 
                                             labels, 
                                             test_size=0.3, 
                                             random_state=42)
'''

def hfo_rate(patients, hfo_type_name):
    return False
def random_forest(patients, hfo_type_name):
    # Select all that have any elec in 'Hippocampus'
    print('Counting patients for {0} model... ~> {1} patients.'.format(hfo_type_name, len(patients)))

    train_patients, test_patients = segmentate_2(patients, train_p=0.8)
    train_pat_amount = len(train_patients)
    del patients

    if hfo_type_name in ['RonO', 'Fast RonO']:
        feature_names = ['duration', 'freq_pk', 'power_pk',
                         'slow', 'slow_vs', 'slow_angle',
                         'delta', 'delta_vs', 'delta_angle',
                         'theta', 'theta_vs', 'theta_angle',
                         'spindle', 'spindle_vs', 'spindle_angle']
    else:
        feature_names = ['duration', 'freq_pk', 'power_pk',
                         'spike', 'spike_vs', 'spike_angle']

    train_features, train_labels = get_features_and_labels(train_patients, hfo_type_name, feature_names)
    del train_patients
    train_hfo_count = len(train_labels)

    test_features, test_labels = get_features_and_labels(test_patients, hfo_type_name, feature_names)
    del test_patients

    #Can't use this because we are splitting by patients first
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



    '''

    X = dataset.iloc[:, 0:4].values
    y = dataset.iloc[:, 4].values

    sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    '''
    train_features = pd.DataFrame(train_features)
    train_features = np.array(train_features)
    train_labels = pd.DataFrame(train_labels)
    train_labels = np.array(train_labels)

    test_features = pd.DataFrame(test_features)
    test_features = np.array(test_features)
    test_labels = pd.DataFrame(test_labels)
    test_labels = np.array(test_labels)

    rf = RandomForestClassifier(n_estimators=100,
                                criterion='gini', #'entropy'
                                random_state=42,
                                #max_features=None,
                                bootstrap=True, #Sampling with replacement for each tree
                                n_jobs=-1, #use all available processors
                                # min_samples_split= 0.005,
                                # min_samples_leaf= 0.005,
                                verbose=0,
                                oob_score=False, #Whether to use out-of-bag samples to estimate the generalization accuracy.
                                class_weight="balanced"
                                )

    print('Training {0} model with {1} hfos from {2} patients...'.format(hfo_type_name, train_hfo_count, train_pat_amount))
    rf.fit(train_features, train_labels.ravel())
    del train_features, train_labels

    if DEBUG:
        print('Classes order: {0}'.format(rf.classes_))

    # Predict over test
    rf_predictions = rf.predict(test_features)
    rf_probs = rf.predict_proba(test_features)[:, 1]

    # Results
    print_metrics(test_labels, rf_predictions, rf_probs)
    del test_features, test_labels


'''
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train, :], y.iloc[train, :])
        score = model.score(X.iloc[test, :], y.iloc[test, :])
        scores.append(score)
'''

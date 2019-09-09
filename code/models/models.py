import numpy as np
import pandas as pd
import math as mt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, cross_val_score

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
    print('Segmentation --> train_size = {0} test_size = {1}'.format(train_size, test_size))
    train_set = patients[:train_size]
    test_set = patients[train_size:patient_count]
    return train_set, test_set

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

def run_RonO_Model(patients):
    # Select all that have any elec in 'Hippocampus'
    print('Counting patients for RonO model... value ~ {0}'.format(len(patients)))

    train_patients, test_patients = segmentate_2(patients, train_p=0.8)
    del patients
    # hay forma de que cada arbol sea un paciente?

    feature_names = ['duration', 'freq_pk', 'power_pk',
                     'slow', 'slow_vs', 'slow_angle',
                     'delta', 'delta_vs', 'delta_angle',
                     'theta', 'theta_vs', 'theta_angle',
                     'spindle', 'spindle_vs', 'spindle_angle']

    train_features = []
    train_labels = []
    for p in train_patients:
        for e in p.electrodes:
            for h in e.hfos['RonO']:
                feature_row_i = {}
                for feature_name in feature_names:
                    feature_i_j = h.info[feature_name]
                    feature_row_i[feature_name] = feature_i_j

                train_features.append(feature_row_i)
                train_labels.append({'soz': h.info['soz']})
    del train_patients

    test_features = []
    test_labels = []
    for p in test_patients:
        for e in p.electrodes:
            for h in e.hfos['RonO']:
                feature_row_i = {}
                for feature_name in feature_names:
                    if 'angle' in feature_name or 'vs' in feature_name:
                        feature_row_i['SIN({0})'.format(feature_name)] =  mt.sin(h.info[feature_name])
                        feature_row_i['COS({0})'.format(feature_name)] =  mt.cos(h.info[feature_name])
                    else:
                        feature_i_j = h.info[feature_name]
                        feature_row_i[feature_name] = feature_i_j
                test_features.append(feature_row_i)
                test_labels.append({'soz': h.info['soz']})
    del test_patients


    rf = RandomForestClassifier(n_estimators=1000, random_state=42)

    train_features = pd.DataFrame(train_features)
    train_features = np.array(train_features)
    train_labels = pd.DataFrame(train_labels)
    train_labels = np.array(train_labels)

    test_features = pd.DataFrame(test_features)
    test_features = np.array(test_features)
    test_labels = pd.DataFrame(test_labels)
    test_labels = np.array(test_labels)

    print('Training RonO model...')
    # Train the model on training data
    rf.fit(train_features, train_labels.ravel())
    del train_features, train_labels

    # Use the forest's predict method on the test data
    rf_predictions = rf.predict(test_features)
    # Probabilities for each class
    print('Predictions')
    print(rf_predictions)
    rf_probs = rf.predict_proba(test_features)[:, 1]
    # Calculate roc auc

    print('Probs')
    print(rf_probs)
    roc_value = roc_auc_score(test_labels, rf_probs)

    print('Results:')
    print('\tROC AUC of ---> {0}'.format(roc_value))

    hits = 0
    total = len(test_labels)
    for i in range(len(rf_predictions)):
        if rf_predictions[i] == test_labels[i]:
            hits += 1
    print('\tHitrate of ---> {0}'.format(hits / total))

    # Calculate the absolute errors
    #errors = abs(rf_predictions - test_labels)
    #print('Abs Error of ---> {0}'.format(errors))

    del test_features, test_labels


'''
    scores = []
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    for i, (train, test) in enumerate(kfold.split(X, y)):
        model.fit(X.iloc[train, :], y.iloc[train, :])
        score = model.score(X.iloc[test, :], y.iloc[test, :])
        scores.append(score)
'''

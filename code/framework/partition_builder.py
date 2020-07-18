import math as mt
import random

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, \
    NeighbourhoodCleaningRule
from sklearn.preprocessing import RobustScaler, StandardScaler

from utils import load_json, save_json
from config import VALIDATION_NAMES_BY_LOC_PATH


# Todo create partitionBuilder class in ml module

# Reviewed
def build_patient_sets(target_patients_id, hfo_type_name, patients_dic,
                       location):
    '''


    :param target_patients_id: 'ALL', 'MODEL_PATIENTS', 'VALIDAION_PATIENTS'
    :param hfo_type_name: as name says
    :param patients_dic: pname, Patient
    :param location: location name to do ml
    :return:
    :model_patients: lists of patients to train
    :target_patients: lists of patients to test
    :test_partition: lists of lists, each list is the has patient test names
    in target patients list for a test fold training with all model_pat not
    in that list.
    '''
    print('Building patient sets...')
    model_patients, validation_patients = pull_apart_validation_set(
        patients_dic, location, val_size=0.3)

    target_patients, test_partition = [], []
    if target_patients_id == 'ALL':
        target_patients = [p for p in patients_dic.values()]
        # Train with random bootstrapping cross validation with all patients
        model_patients = target_patients
        # 1000 test lists with indexes of patients in target to test in that
        # iteration  sets randomly generated with bootstrapping
        # each iteration trains with the model patients that aren't in the
        # test list.
        test_partition = get_N_fold_bootstrapping_partition(target_patients,
                                                            N=1000,
                                                            test_size=0.3)
        # test_partition = get_k_fold_random_partition(target_patients, K=10)
    elif 'MODEL_PATIENTS' == target_patients_id:
        # For model training
        target_patients = model_patients
        test_partition = get_N_fold_bootstrapping_partition(target_patients,
                                                            N=1000,
                                                            test_size=0.3)

        # test_partition = get_k_fold_random_partition(target_patients, K=10)
    elif 'VALIDATION_PATIENTS' in target_patients_id:
        # For model validation
        target_patients = validation_patients
        test_partition = [[p.id for p in validation_patients]]

    return model_patients, target_patients, test_partition


# Reviewed
# Descripción: Si es la primera vez que ve la location guarda en un json.
# Tomamos test_size como proporcion de pacientes para test, calculando la
# proporcion por separado para pacientes con y sin epilepsia en location
# para evitar validar con una sola clase. Elegimos nombres random de
# pacientes mezclando indices
# retorna dos listas de pacientes, los que se usan para entrenar y
# testear (model_patients) y los que vamos a usar despues para validar
def pull_apart_validation_set(patients_dic, location, val_size=0.3):
    print('Pulling apart validation patient set...')
    # Loads predefined validation patient names randomly selected for location
    names_by_loc = load_json(VALIDATION_NAMES_BY_LOC_PATH)
    if location not in names_by_loc.keys():  # first time, we set validation set
        # Build validation set
        names_by_loc[location] = dict()
        # We will take val_size of each soz_patients and healthy patients to
        # avoid the possibility of validating with just one class.
        soz_pat, healthy_pat = [], []
        for p_name, p in patients_dic.items():
            if p.has_epilepsy_in_loc(location):
                soz_pat.append(p)
            else:
                healthy_pat.append(p)

        # Gets a set of the names of 0.3 random patients of the list
        validation_names_soz = get_N_fold_bootstrapping_partition(soz_pat,
                                                                  N=1,
                                                                  test_size=val_size)[
            0]
        # Gets a set of the names of 0.3 random patients of the list
        validation_names_healthy = \
            get_N_fold_bootstrapping_partition(healthy_pat,
                                               N=1,
                                               test_size=val_size)[0]
        validation_names = validation_names_soz + validation_names_healthy
        names_by_loc[location] = validation_names
        save_json(names_by_loc, VALIDATION_NAMES_BY_LOC_PATH)  # Update json

    else:
        validation_names = names_by_loc[location]
    print('Validation patient names in {l}'.format(l=location,
                                                   v=validation_names))
    model_patients = [p for p_name, p in patients_dic.items() if p_name not
                      in validation_names]
    validation_patients = [p for p_name, p in patients_dic.items() if p_name
                           in validation_names]

    return model_patients, validation_patients


# Reviewed
def get_k_fold_random_partition(target_patients, K):
    partition_names = [[] for i in range(K)]
    idx = list(range(len(target_patients)))  # [1,2...N]
    random.shuffle(idx)
    for i in range(len(target_patients)):
        partition_names[i % K].append(target_patients[idx[i]].id)
    return partition_names


# Reviewed
# Creates N lists of indexes refering patients in target_pat to test in a
# iteration. test_size is the proportion of target patients to test in each iter
# returns as list of lists of patient names
def get_N_fold_bootstrapping_partition(target_patients, N, test_size=0.3):
    partition_names = []
    idx = list(range(len(target_patients)))  # [0,1,2...len-1]
    # Mezcla N veces y forma una lista con los primeros len(patients) *
    # test_size indices despues de mezclar
    for i in range(N):
        random.shuffle(idx)
        index_list = idx[:int((len(target_patients) * test_size))]
        names_list = [target_patients[i].id for i in index_list]
        partition_names.append(names_list)

    return partition_names


# Reviewed
# TODO experiment with sacaler and resampling with SMOTE
def build_folds(hfo_type_name, model_patients, target_patients, test_partition):
    target_patients_dic = {p.id: p for p in target_patients}
    # Observation: this above maintains target_patient order
    target_patient_names = [p.id for p in target_patients]
    field_names = ml_field_names(hfo_type_name)
    fold = 0
    folds = []
    for p_names in test_partition:  # p_names es una lista de pacientes en test
        fold += 1
        print('Building fold {f}...'.format(f=fold))
        train_patients = [p for p in model_patients if p.id not in p_names]
        # Los pacientes a testear estan en el orden de p_names
        test_patients = [target_patients_dic[name] for name in p_names]

        # Orden de indices por orden de p_names
        target_pat_idx = [target_patient_names.index(name) for name in p_names]

        train_features_dic, train_labels = get_features_and_labels(
            train_patients, hfo_type_name, field_names)
        test_features_dic, test_labels = get_features_and_labels(test_patients,
                                                                 hfo_type_name,
                                                                 field_names)
        # Notar que test labels esta transformado en el orden de p_names ya
        # que sigue el orden de test_patients, por lo que es importante que
        # target_pat_idx tambien se construya iterando sobre p_names para
        # tener correcto el indice para guardar las predicciones luego
        train_features_pd = pd.DataFrame(train_features_dic)
        test_features_pd = pd.DataFrame(test_features_dic)
        feature_names = test_features_pd.columns  # adds sin and cos for PAC
        train_features, test_features = train_features_pd.values, test_features_pd.values

        # TODO review scaler
        # scaler = RobustScaler()  # Scale features using statistics that are
        # robust to outliers.
        # scaler = StandardScaler()
        # train_features = scaler.fit_transform(train_features)
        # test_features = scaler.transform(test_features)

        # Resampling and balancing classes
        # train_features, train_labels = balance_samples(train_features, train_labels)
        # train_features = scaler.fit_transform(train_features)
        # test_features = scaler.transform(test_features)

        folds.append({
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'target_pat_idx': target_pat_idx,
            'feature_names': feature_names
        })
    return folds


# Features del modelo de ml segun el hfo_type
def ml_field_names(hfo_type_name, include_coords=False):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        field_names = ['duration',
                       'power_av',
                       'freq_av',
                       'slow_angle',
                       'delta_angle',
                       'theta_angle',
                       'spindle_angle']
    else:
        field_names = ['duration',
                       'power_pk', 'power_av',
                       'freq_pk', 'freq_av',
                       'spike_angle']
    if include_coords:
        for c in ['x', 'y', 'z']:
            field_names.append(c)

    return field_names


# TODO probar la varianza o mean del feature en el electrodo como nuevo feature
def get_features_and_labels(patients, hfo_type_name, field_names):
    features = []
    labels = []
    for p in patients:
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                feature_row_i = {}
                for feature_name in field_names:
                    if 'angle' in feature_name or 'vs' in feature_name:
                        feature_row_i['SIN({0})'.format(feature_name)] = mt.sin(
                            h.info[feature_name])
                        feature_row_i['COS({0})'.format(feature_name)] = mt.cos(
                            h.info[feature_name])
                    else:
                        feature_row_i[feature_name] = h.info[feature_name]
                features.append(feature_row_i)
                labels.append(h.info['soz'])

    return features, np.array(labels)  # returns dict, np.array, list


# Reviewed
def analize_fold_balance(train_labels):
    positive_class = 0
    negative_class = 0
    tot = len(train_labels)
    i = 0
    for l in train_labels:
        i += 1
        if l:
            positive_class += 1
        else:
            negative_class += 1
    print('Fold balance: Positive: {0} Negative: {1}. Count: {2}'.format(
        round(positive_class / tot, 2), round(negative_class / tot, 2), tot))


# Reviewed
def balance_samples(features, labels):
    prev_count = len(features)
    analize_fold_balance(labels)

    print('Performing resample with SMOTETomek...')
    print('Original train hfo count : {0}'.format(prev_count))

    # smt = RepeatedEditedNearestNeighbours( n_jobs=-1)
    # smt = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=3, n_jobs=-1)
    smt = SMOTETomek(sampling_strategy=1, random_state=42, n_jobs=4)
    features, labels = smt.fit_resample(features, labels)
    post_count = len(features)

    print('{0} instances after SMOTE...'.format(post_count))
    return features, labels


# Reviewed
def patients_with_more_than(count, patients_dic, hfo_type_name):
    with_hfos = dict()
    without_hfos = dict()
    for p_name, p in patients_dic.items():
        if sum([len(e.events[hfo_type_name]) for e in
                p.electrodes]) > count:  # if has more than count hfos passes
            with_hfos[p_name] = p
        else:
            without_hfos[p_name] = p
    return with_hfos, without_hfos

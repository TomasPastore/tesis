import copy
import itertools
import math as mt
import random

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import RepeatedEditedNearestNeighbours, NeighbourhoodCleaningRule
from sklearn.preprocessing import RobustScaler, StandardScaler


# Filtra pacientes que cumplen un threshold de cantidad de eventos
from config import hip_rons_validation_names
#Todo shouldn't this be inside ml module? 

def patients_with_more_than(count, hfo_type_name, patients_dic):
    with_hfos = dict()
    without_hfos = dict()
    for p_name, p in patients_dic.items():
        if sum([len(e.events[hfo_type_name]) for e in p.electrodes]) > count:  # si tiene mas de count hfos
            with_hfos[p_name] = p
        else:
            without_hfos[p_name] = p
    return with_hfos, without_hfos


def get_random_partition(target_patients, K):
    partition_names = [[] for i in range(K)]
    idx = list(range(len(target_patients)))
    random.shuffle(idx)
    for i in range(len(target_patients)):
        partition_names[i%K].append(target_patients[idx[i]].id)
    return partition_names

def build_patient_sets(target, hfo_type_name, patients_dic):
    print('\nBuilding patient sets...')
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)

    if 'model_pat' in target and 'validation_pat' in target:
        target_patients = [p for p in patients_dic.values()]
        model_patients = target_patients
        #test_partition = get_balanced_partition(target_patients, hfo_type_name, K=3, method='deterministic')
        test_partition = get_random_partition(target_patients, K=3)

    elif 'model_pat' in target:
        target_patients = model_patients
        #test_partition = get_balanced_partition(target_patients, hfo_type_name, K=3, method='deterministic')
        test_partition = get_random_partition(target_patients, K=3)
    elif 'validation_pat' in target:
        target_patients = validation_patients
        test_partition = [[p.id for p in validation_patients]]

    return model_patients, target_patients, test_partition

def get_balanced_partition(patients, hfo_type_name, K=4, method='deterministic'):
    count_by_pat = []
    balance_by_pat = []
    phfo_per_pat = {}
    for p in patients:
        count_by_pat.append((p.id, sum([len(e.events[hfo_type_name]) for e in p.electrodes])))
        balance_by_pat.append((p.id, p.get_class_balance(hfo_type_name)))
        phfos_per_elec = [(e.pevt_count[hfo_type_name], len(e.events[hfo_type_name])) for e in p.electrodes]
        #print(phfos_per_elec)
        elec_flags = [e.soz for e in p.electrodes]
        #print(elec_flags)
        phfo_per_pat[p.id] = (sum([t[0] for t in phfos_per_elec]), sum([t[1] for t in phfos_per_elec]), any(elec_flags))
    # count_by_pat is a [(pat_name, hfo_count)]
    count_by_pat.sort(key=lambda tup: tup[1])
    #print('hfo count by patient')
    #print(count_by_pat)
    phfo_per_pat_list = []
    for pid, s in count_by_pat:
        phfo_per_pat_list.append(phfo_per_pat[pid])
    #print(phfo_per_pat_list)
    balance_by_pat.sort(key=lambda tup:  (tup[1][0] / tup[1][2] - tup[1][1] / tup[1][2]) if tup[1][2] !=0 else 0)

    hfo_ordered_names = [name for name, h_count in count_by_pat]
    cbp_ordered_names = [name for name, _ in balance_by_pat]

    # mantiene los hfo que hay en cada particion
    partition_hfos = [0 for _ in range(K)]
    partition_names = [[] for _ in range(K)]
    # mantiene un arreglo de tuplas (negative_count, positive_count, tot_count) para cada grupo,
    # las tuplas corresponden a la proporcion de los pacientes
    partition_bperc = [[] for _ in range(K)]

    print('\nCreating patients partition...')
    print('Total hfo count: {0}'.format(sum([t[1] for t in count_by_pat])))
    print('Mean hfo count per patient: {0}'.format(np.mean([t[1] for t in count_by_pat])))
    print('Standard deviation: {0}'.format(np.std([t[1] for t in count_by_pat])))
    print('Patient hfo count: {0}'.format(count_by_pat))

    if method == 'random':
        random.shuffle(count_by_pat)
        for name, count in count_by_pat:
            best_place = 0
            for candidate_group_idx in range(K):
                current_groups = copy.deepcopy(partition_hfos)
                current_groups[best_place] += count
                candidate_groups = copy.deepcopy(partition_hfos)
                candidate_groups[candidate_group_idx] += count
                if np.std(candidate_groups) < np.std(current_groups):
                    best_place = candidate_group_idx

            partition_hfos[best_place] += count
            partition_names[best_place].append(name)
    elif method == 'deterministic':
        # Add patients at the end with 0 count to have a multiple of K, then I remove them
        fake_patients = len(count_by_pat) % K
        for f in range(fake_patients):
            count_by_pat.append(('fake', 0))

        for i in range(len(count_by_pat) // K):
            s_assert = sum(partition_hfos)
            k_candidates = count_by_pat[i * K: (i + 1) * K]
            rel_idxs = [idx for idx in range(K)]
            permutations = itertools.permutations(rel_idxs)
            best_permutation = rel_idxs
            for p in permutations:
                current_groups = [partition_hfos[k] + k_candidates[best_permutation[k]][1] for k in range(K)]
                candidate_groups = [partition_hfos[k] + k_candidates[p[k]][1] for k in range(K)]
                if np.std(candidate_groups) < np.std(current_groups):
                    best_permutation = p

            assert (sum(partition_hfos) == s_assert)
            for k in range(K):
                partition_names[k].append(k_candidates[best_permutation[k]][0])
                partition_hfos[k] += k_candidates[best_permutation[k]][1]

        for p in partition_names:
            try:
                p.remove('fake')
            except ValueError:
                pass

    elif method == 'balanced':
        # Add patients at the end with 0 count to have a multiple of K, then I remove them
        fake_patients = len(balance_by_pat) % K
        for f in range(fake_patients):
            balance_by_pat.append(('fake', 0))

        for i in range(len(balance_by_pat) // K):
            k_candidates = balance_by_pat[i * K: (i + 1) * K]
            k_candidates_hfos = [count_by_pat[cbp_ordered_names.index(id)][1] for id, _ in
                                 balance_by_pat[i * K: (i + 1) * K]]

            rel_idxs = [idx for idx in range(K)]
            permutations = itertools.permutations(rel_idxs)
            best_permutation = rel_idxs
            for p in permutations:
                current_groups = [partition_bperc[k] + [k_candidates[best_permutation[k]][1]] for k in range(K)]
                candidate_groups = [partition_bperc[k] + [k_candidates[p[k]][1]] for k in range(K)]

                crit_1 = abs(np.mean([np.mean([nc - tc for nc, tc, tot_c in candidate_groups[k]]) for k in range(K)])) < \
                         abs(np.mean([np.mean([nc - tc for nc, tc, tot_c in current_groups[k]]) for k in range(K)]))

                crit_2 = abs(np.mean([(nc / tot_c - tc / tot_c)  if tot_c!=0 else 0 for nc, tc, tot_c in colapsar(candidate_groups)])) < \
                         abs(np.mean([(nc / tot_c - tc / tot_c)  if tot_c!=0 else 0 for nc, tc, tot_c in colapsar(current_groups)]))

                crit = crit_2
                if crit:
                    best_permutation = p

            for k in range(K):
                partition_names[k].append(k_candidates[best_permutation[k]][0])
                partition_bperc[k] = partition_bperc[k] + [k_candidates[best_permutation[k]][1]]
                partition_hfos[k] += k_candidates_hfos[best_permutation[k]]

        for p in partition_names:
            try:
                p.remove('fake')
            except ValueError:
                pass

    else:
        raise RuntimeError('Unknown partitioning method name')

    print('Patient partition created...')
    print('Groups hfo count: {0}'.format(partition_hfos))
    print('Mean hfo count per group: {0}'.format(np.mean(partition_hfos)))
    print('Standard deviation among groups: {0}\n'.format(np.std(partition_hfos)))
    #print('Indexes by group...')
    #orig_pat_order = [p.id for p in patients]
    #for k in range(K):
    #    print('\t Group {k}: {idxs}'.format(k=k, idxs=[orig_pat_order.index(name) for name in partition_names[k]]))

    return partition_names


def colapsar(partition):
    colapsed = []
    for p in partition:
        n_count = 0
        t_count = 0
        tot_count = 0
        for nc, tc, tot_c in p:
            t_count += tc
            n_count += nc
            tot_count += tc
        colapsed.append((n_count, t_count, tot_count))
    return colapsed


def pull_apart_validation_set(patients_dic, hfo_type_name):
    print('Building validation patient set...')
    patients = [p for p in patients_dic.values()]

    predefined = True
    if predefined: #'Hippocampus RonS'
        print('Predefined validation names for RonS in Hippocampus')
        #random.shuffle(patients)
        #validation_patients = patients[:len(patients) // 3]
        #model_patients = patients[len(patients)//3 : len(patients)]
        validation_names =  hip_rons_validation_names
        validation_patients = [patients_dic[name] for name in validation_names]
        model_patients = [p for name, p in patients_dic.items() if name not in validation_names]
    else:
        partition_names = get_balanced_partition(patients, hfo_type_name, K=3, method='deterministic')
        validation_patients = [patients_dic[name] for name in partition_names[0]]
        model_patients = [p for p_name, p in patients_dic.items() if p_name not in partition_names[0]]

    print('Validation patient names {0}'.format([p.id for p in validation_patients]))
    return model_patients, validation_patients


# Features del modelo de ml segun el hfo_type
def ml_field_names(hfo_type_name, include_coords=False):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        field_names = ['duration',
                       'power_pk', 'power_av',
                       'freq_pk', 'freq_av',
                       'slow_angle',
                       'delta_angle',
                       'theta_angle',
                       'spindle_angle']  # 'slow', 'delta','theta', 'spindle',
    else:
        field_names = ['duration',
                       'power_pk', 'power_av',
                       'freq_pk','freq_av',
                       'spike_angle']
    if include_coords:
        for c in ['x', 'y', 'z']:
            field_names.append(c)

    return field_names


def get_features_and_labels(patients, hfo_type_name, field_names):
    features = []
    labels = []
    for p in patients:
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                feature_row_i = {}
                for feature_name in field_names:
                    if 'angle' in feature_name or 'vs' in feature_name:
                        feature_row_i['SIN({0})'.format(feature_name)] = mt.sin(h.info[feature_name])
                        feature_row_i['COS({0})'.format(feature_name)] = mt.cos(h.info[feature_name])
                    else:
                        feature_row_i[feature_name] = h.info[feature_name]
                features.append(feature_row_i)
                labels.append(h.info['soz'])

    return features, np.array(labels)  # returns dict, np.array, list


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


def balance_samples(features, labels):
    prev_count = len(features)
    analize_fold_balance(labels)

    print('Performing resample with SMOTETomek...')
    print('Original train hfo count : {0}'.format(prev_count))

    #smt = RepeatedEditedNearestNeighbours( n_jobs=-1)
    #smt = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=3, n_jobs=-1)
    smt = SMOTETomek(sampling_strategy=1, random_state=42, n_jobs=4)
    features, labels = smt.fit_resample(features, labels)
    post_count = len(features)

    print('{0} instances after SMOTE...'.format(post_count))
    return features, labels


def build_folds(hfo_type_name, model_patients, target_patients, test_partition):
    target_patients_dic = {p.id:p for p in target_patients}
    target_patient_names = [p.id for p in target_patients]  # Obs mantiene el orden de target_patients
    field_names = ml_field_names(hfo_type_name)
    fold = 0
    folds = []
    for p_names in test_partition:
        fold += 1
        print('Building fold {f}...'.format(f=fold))
        train_patients = [p for p in model_patients if p.id not in p_names]
        test_patients = [target_patients_dic[name] for name in p_names]
        target_pat_idx = [target_patient_names.index(name) for name in p_names]

        train_features_dic, train_labels = get_features_and_labels(train_patients, hfo_type_name, field_names)
        test_features_dic, test_labels = get_features_and_labels(test_patients, hfo_type_name, field_names)
        train_features_pd = pd.DataFrame(train_features_dic)
        test_features_pd = pd.DataFrame(test_features_dic)
        feature_names = test_features_pd.columns
        train_features, test_features = train_features_pd.values, test_features_pd.values

        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        #scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        #train_features, train_labels = balance_samples(train_features, train_labels)
        #train_features = scaler.fit_transform(train_features)
        #test_features = scaler.transform(test_features)
        folds.append({
            'train_features': train_features,
            'train_labels': train_labels,
            'test_features': test_features,
            'test_labels': test_labels,
            'target_pat_idx': target_pat_idx,
            'feature_names': feature_names
        })
    return folds


'''
def encode_locations(train_features_pd, test_features_pd, loc_names=['loc1', 'loc2', 'loc3', 'loc4', 'loc5']):
    merged_pd = pd.concat([train_features_pd, test_features_pd], axis=0)
    locations = pd.get_dummies(merged_pd, columns=loc_names)
    for name in loc_names:
        locations = locations.drop(['{loc}_empty'.format(loc=name)], axis=1)
        merged_pd = merged_pd.drop(['{loc}'.format(loc=name)], axis=1)

    merged_pd = pd.concat([merged_pd, locations], axis=1)
    train_features_pd = merged_pd[:len(train_features_pd)]
    test_features_pd = merged_pd[len(train_features_pd):]

    return train_features_pd, test_features_pd
'''

'''
def get_feature(i, j, features, name):
    if 'SIN' in name:
        return name[4:-1], mt.asin(features[i][j])  # removes 'SIN()'
    elif 'COS' in name:
        return name[4:-1], mt.acos(features[i][j])  # removes 'COS()'
    else:
        return name, features[i][j]
'''

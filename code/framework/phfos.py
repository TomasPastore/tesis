import copy
import getpass
import itertools
import math as mt
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from config import models_to_run, models_to_run_obj, EVENT_TYPES
from metrics import print_metrics
from models import xgboost
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV

#######################      Patients Partition     #############################

#TODO hacer modulo patients_partition
#Preprocesado con lista de pacientes para armar bien la particion de cross validation de manera 'balanceada'

#TODO
def pathological_analisis():
    #pevent_p por electrodo en histograma
    fevent_p_by_elec = [] #porcentaje de eventos fisiologicos por electrodo
    pevent_p_by_elec = [] #porcentaje de eventos patologicos por electrodo
    phfo_counts_by_elec = [] #cantidad ''
    #usar rate data para ver otras cosas como pevents_p promedio
    #TODO scatter fhfo % vs phfo % por electrodo


#Filtra pacientes que cumplen un threshold de cantidad de eventos
# TODO ese umbral de 100 se podria mejorar estudiando la distribucion de hfo_count
def patients_with_hfos(patients_dic, hfo_type_name):
    with_hfos = dict()
    without_hfos = dict()
    for p_name, p in patients_dic.items():
        if sum([len(e.hfos[hfo_type_name]) for e in p.electrodes]) > 100: #si tiene mas de 100 hfo
            with_hfos[p_name] = p
        else:
            without_hfos[p_name] = p
    return with_hfos, without_hfos


def pull_apart_validation_set(patients_dic, hfo_type_name):
    patients = [p for p in patients_dic.values()]
    partition_names = get_balanced_partition(patients, hfo_type_name, K=4, method='deterministic')
    print('Validation names {0}'.format(partition_names[0]))
    validation_patients = [patients_dic[name] for name in partition_names[0]]
    model_patients = [p for p_name, p in patients_dic.items() if p_name not in partition_names[0]]
    return model_patients, validation_patients


#TODO comprobar que los de validacion son siempre los mismos
def get_balanced_partition(patients, hfo_type_name, K=4, method='deterministic'):
    count_by_pat = []
    balance_by_pat = []
    for p in patients:
        count_by_pat.append((p.id, sum([len(e.hfos[hfo_type_name]) for e in p.electrodes])))
        balance_by_pat.append((p.id, p.get_class_balance(hfo_type_name)))

    # count_by_pat is a [(pat_name, hfo_count)]
    count_by_pat.sort(key=lambda tup: tup[1])
    balance_by_pat.sort(key=lambda tup: (tup[1][0]/tup[1][2])- (tup[1][1]/tup[1][2]))

    hfo_ordered_names = [name for name, h_count in count_by_pat]
    cbp_ordered_names = [name for name, _ in balance_by_pat]

    # mantiene los hfo que hay en cada particion
    partition_hfos = [0 for _ in range(K)]
    partition_names = [[] for _ in range(K)]
    # mantiene un arreglo de tuplas (negative_count, positive_count, tot_count) para cada grupo,
    # las tuplas corresponden a la proporcion de los pacientes
    partition_bperc = [[] for _ in range(K)]

    print('Total hfo count: {0}'.format(sum([t[1] for t in count_by_pat])))
    print('Mean hfo count per patient: {0}'.format(np.mean([t[1] for t in count_by_pat])))
    print('Standard deviation: {0}'.format(np.std([t[1] for t in count_by_pat])))
    print('Values {0}'.format(count_by_pat))

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

    elif method == 'balance_classes':
        # Add patients at the end with 0 count to have a multiple of K, then I remove them
        fake_patients = len(balance_by_pat) % K
        for f in range(fake_patients):
            balance_by_pat.append(('fake', 0))

        for i in range(len(balance_by_pat) // K):
            k_candidates = balance_by_pat[i * K: (i + 1) * K]
            k_candidates_hfos = [count_by_pat[cbp_ordered_names.index(id)][1] for id, _ in balance_by_pat[i * K: (i + 1) * K] ]

            rel_idxs = [idx for idx in range(K)]
            permutations = itertools.permutations(rel_idxs)
            best_permutation = rel_idxs
            for p in permutations:
                current_groups = [partition_bperc[k] + [k_candidates[best_permutation[k]][1]] for k in range(K)]
                candidate_groups = [partition_bperc[k] + [k_candidates[p[k]][1]] for k in range(K)]

                crit_1= abs( np.mean([ np.mean( [nc - tc for nc, tc, tot_c in candidate_groups[k]] ) for k in range(K) ] ) ) < \
                        abs( np.mean([ np.mean( [nc - tc for nc, tc, tot_c in current_groups[k]] ) for k in range(K) ] ) )

                crit_2 = abs(np.mean([(nc/tot_c) - (tc/tot_c) for nc, tc, tot_c in colapsar(candidate_groups)])) < \
                         abs(np.mean([(nc/tot_c) - (tc/tot_c) for nc, tc, tot_c in colapsar(current_groups) ]))

                crit = crit_2
                if  crit:
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

    print('Partition analysis...')
    print('Groups hfo count: {0}'.format(partition_hfos))
    print('Mean hfo count per group: {0}'.format(np.mean(partition_hfos)))
    print('Standard deviation among groups: {0}'.format(np.std(partition_hfos)))
    print('Indexes by group...')
    orig_pat_order = [p.id for p in patients]
    for k in range(K):
        print('\t Group {k}: {idxs}'.format(k=k, idxs=[orig_pat_order.index(name) for name in partition_names[k]]))

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


#/################      Patients Partition    ########################/

##################      Features              #########################

#Features del modelo de ml segun el hfo_type
def ml_field_names(hfo_type_name):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        field_names = ['duration', 'freq_av', 'power_av',
                        'slow_angle',
                        'delta_angle',
                        'theta_angle',
                        'spindle_angle', 'x', 'y', 'z']  # 'slow', 'delta','theta', 'spindle',
    else:
        field_names = ['duration', 'power_av', 'freq_av',
                       'spike_angle', 'x', 'y', 'z']  # ,' 'age', 'spike',,'spike_vs'freq_pk', 'power_pk'

    return field_names

def get_features_and_labels(patients, hfo_type_name, field_names):
    features = []
    labels = []
    for p in patients:
        for e in p.electrodes:
            for h in e.hfos[hfo_type_name]:
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

def balance_samples(features, labels):
    prev_count = len(features)
    print('Original train hfo count : {0}'.format(prev_count))
    print('Performing resample with SMOTETomek...')
    smt = SMOTETomek(sampling_strategy=1, random_state=42, n_jobs=4)
    #smt = NeighbourhoodCleaningRule(sampling_strategy='majority', n_neighbors=3, n_jobs=-1)
    features, labels = smt.fit_resample(features, labels)
    post_count = len(features)
    print('{0} instances after SMOTE...'.format(post_count))
    analize_balance(labels)

    return features, labels


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
#/################      Features             ########################/

##################      Resultados           #######################

# Guarda los resultados del fold en la estructura global de patients
def save_prediction(clf_preds, clf_probs, patients, test_pat_idx, test_labels, hfo_type_name, model):
    i = 0
    for t in test_pat_idx:
        for e in patients[t].electrodes:
            for h in e.hfos[hfo_type_name]:
                assert (h.info['prediction'][model] == 0)
                h.info['prediction'][model] = int(clf_preds[i])
                if clf_probs is not None:
                    assert (h.info['proba'][model] == 0)
                    h.info['proba'][model] = clf_probs[i]
                # asserts that the hfo is being selected correctly for result i
                assert (test_labels[i] == h.info['soz'])
                i += 1


# ROCs

def analize_balance(train_labels):
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
    print('Percentages in analize partition: Positive: {0} Negative: {1}. Count: {2}'.format(round(positive_class / tot, 2)))

def axes_by_model(plt):
    subplot_count = len(models_to_run) * 2
    if subplot_count == 2:
        rows = 2
        cols = 1
    elif subplot_count == 4:
        rows = 2
        cols = 2
    elif subplot_count == 6:
        rows = 2
        cols = 3
    else:
        raise RuntimeError('Subplot count not implemented')
    axes = {}
    subplot_index = 1
    for m in models_to_run:
        if m not in axes.keys():
            axes[m] = {}
        axes[m]['ROC'] = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        i += 1
    for m in models_to_run:
        axes[m]['PRE_REC'] = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        i += 1
    return axes


def plot_roc_fold(fpr, tpr, model_name, plot_axe, mean_fpr, tprs, aucs, fold):
    curve_kind = 'ROC '
    interp_tpr = interp(mean_fpr, fpr, tpr)
    tprs[model_name].append(interp_tpr)
    tprs[model_name][-1][0] = 0.0

    roc_auc = auc(fpr, tpr)
    aucs[model_name][curve_kind].append(roc_auc)
    plot_axe[model_name][curve_kind].plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc))

    return tprs, aucs


def average_ROCs(model_name, plot_axe, mean_fpr, tprs, aucs):
    curve_kind = 'ROC'
    plot_axe[model_name][curve_kind].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs[model_name], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[model_name][curve_kind])
    plot_axe[model_name][curve_kind].plot(mean_fpr, mean_tpr, color='b',
                              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                              lw=2, alpha=.8)

    std_tpr = np.std(tprs[model_name], axis=0)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    plot_axe[model_name][curve_kind].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                      label=r'$\pm$ 1 std. dev.')

    plot_axe[model_name][curve_kind].set_xlim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_ylim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_xlabel('False Positive Rate')
    plot_axe[model_name][curve_kind].set_ylabel('True Positive Rate')
    plot_axe[model_name][curve_kind].set_title('{0} ROC curves'.format(model_name))
    plot_axe[model_name][curve_kind].legend(loc="lower right")

#PRE REC

def plot_pre_rec_fold(recall, precision, model_name, plot_axe, mean_recall, prec, aucs, fold, average_precision):
    curve_kind = 'PRE_REC'
    interp_prec = interp(mean_recall, recall, precision)
    prec[model_name].append(interp_prec)
    prec[model_name][-1][0] = 1
    auc_v = auc(recall, precision)
    aucs[model_name][curve_kind].append(auc_v)
    plot_axe[model_name][curve_kind].plot(recall, precision, lw=1, alpha=0.3,
                            label='%s fold %d (AUC = %0.2f. AP = %0.2f.)' % (fold, auc_v, average_precision))

    return prec, aucs


def average_pre_rec(model_name, plot_axe, mean_recall, prec, aucs, aps):
    curve_kind = 'PRE_REC'
    mean_prec = np.mean(prec[model_name], axis=0)
    mean_prec[-1] = 0 #TODO revisar
    mean_auc = auc(mean_recall, mean_prec)
    std_auc = np.std(aucs[model_name][curve_kind])
    mean_ap = np.mean(aps[model_name])
    std_ap = np.std(aps[model_name])

    plot_axe[model_name][curve_kind].plot(mean_recall, mean_prec, color='b',
                              label=r'Mean PRE_REC (AUC = %0.2f $\pm$ %0.2f. AP = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc, mean_ap, std_ap),
                              lw=2, alpha=.8)

    std_prec = np.std(prec[model_name], axis=0)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    plot_axe[model_name][curve_kind].fill_between(mean_recall, prec_lower, prec_upper, color='grey', alpha=.2,
                                      label=r'$\pm$ 1 std. dev.')

    plot_axe[model_name][curve_kind].set_xlim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_ylim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_xlabel('Recall')
    plot_axe[model_name][curve_kind].set_ylabel('Precision')
    plot_axe[model_name][curve_kind].set_title('{0} Precision-Recall curves'.format(model_name))
    plot_axe[model_name][curve_kind].legend(loc="lower right")

#Junta todos los resultados de las particiones para analizar, tiene menos error la muestra por ser más grande
def gather_folds(model_name, hfo_type_name, model_patients):
    labels = []
    preds = []
    probs = []
    for p in model_patients:
        for e in p.electrodes:
            for h in e.hfos[hfo_type_name]:
                labels.append(h.info['soz'])
                # Checked that classes are [False, True] order
                preds.append(h.info['prediction'][model_name])
                probs.append(h.info['proba'][model_name])

    return labels, preds, probs
##################      Resultados           #######################


################  EXPERIMENTACION ML ##############################

def compare_phfo_models(hfo_type_name, loc_name, patients_dic):
    print('Comparing phfo models for hfo type: {0} in {1}... '.format(hfo_type_name, loc_name))
    patients_dic, _ = patients_with_hfos(patients_dic, hfo_type_name)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    model_patient_names = [p.id for p in model_patients]  # Obs mantiene el orden de model_patients

    K = 3
    test_partition = get_balanced_partition(model_patients, hfo_type_name, K=K, method='deterministic')
    field_names = ml_field_names(hfo_type_name)
    fold = 0

    fig = plt.figure(0,figsize=(10, 5))
    fig.suptitle('Phfo models in {location}'.format(location=loc_name), fontsize=16)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = {m: [] for m in models_to_run}
    #f_thresholds = {m: [] for m in models_to_run}
    aucs = {m: [] for m in models_to_run}
    aps = {m: [] for m in models_to_run}
    plot_axe = axes_by_model(plt)
    for p_names in test_partition:  # [[p.id for p in validation_patients]]
        fold += 1
        print('Running fold {f}...'.format(f=fold))
        train_patients = [p for p in model_patients if p.id not in p_names]
        test_patients = [patients_dic[name] for name in p_names]
        test_pat_idx = [model_patient_names.index(name) for name in p_names]
        train_features, train_labels = get_features_and_labels(train_patients, hfo_type_name, field_names)
        test_features, test_labels = get_features_and_labels(test_patients, hfo_type_name, field_names)

        train_features_pd = pd.DataFrame(train_features)
        test_features_pd = pd.DataFrame(test_features)
        feature_names = test_features_pd.columns
        train_features, test_features = train_features_pd.values, test_features_pd.values
        analize_balance(train_labels)
        # Se balancea antes o despues de escalar? probar como parte del exp
        # train_features, train_labels = balance_samples(train_features, train_labels)

        # probar que scaler uso como parte del exp
        #scaler = StandardScaler()
        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        for model_name, model_func in zip(models_to_run, models_to_run_obj):
            clf_preds, clf_probs = model_func(train_features, train_labels, test_features,
                                              feature_list=feature_names, hfo_type_name=hfo_type_name)
            save_prediction(clf_preds, clf_probs, model_patients, test_pat_idx, test_labels,
                            hfo_type_name, model=model_name)

            # ROC
            fpr, tpr, thresholds = roc_curve(test_labels, clf_probs)
            tprs = plot_roc_fold(fpr, tpr, model_name, plot_axe, mean_fpr, tprs, aucs, fold)

            # PRE REC
            precision, recall, thresholds = precision_recall_curve(test_labels, clf_probs)
            precision = np.array(list(reversed(list(precision))))
            recall = np.array(list(reversed(list(recall))))
            thesholds = np.array(list(reversed(list(thresholds))))
            ap = average_precision_score(test_labels, clf_probs)
            aps[model_name].append(ap)
            prec = plot_pre_rec_fold(recall, precision, model_name, plot_axe, mean_recall, prec, aucs, fold,
                                     average_precision=ap)

    for model_name in models_to_run:
        average_ROCs(model_name, plot_axe, mean_fpr, tprs, aucs)
        average_pre_rec(model_name, plot_axe, mean_recall, prec, aucs, aps)

        # Esto es opcional
        labels, preds, probs = gather_folds(model_name, hfo_type_name, model_patients)
        print_metrics(model_name, hfo_type_name, labels, preds, probs)

    plt.savefig('/home/{user}/{type}_phfo_model_comparison_{loc}.png'.format(user=getpass.getuser(), type=hfo_type_name,
                                                                             loc=loc_name), format='png')
    plt.show()


################################ PARAM TUNING START ########################################


def param_tuning(hfo_type_name, patients_dic):
    print('Analizying models for hfo type: {0} in {1}... '.format(hfo_type_name, 'Hippocampus'))
    patients_dic, _ = patients_with_hfos(patients_dic, hfo_type_name)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    model_patient_names = [p.id for p in model_patients]  # Obs mantiene el orden de model_patients
    field_names = ml_field_names(hfo_type_name)
    test_partition = get_balanced_partition(model_patients, hfo_type_name, K=4, method='balance_classes')
    column_names = []
    train_data = []
    labels = []
    partition_ranges = []
    i = 0
    for p_names in test_partition:
        test_patients = [patients_dic[name] for name in p_names]
        x, y = get_features_and_labels(test_patients, hfo_type_name, field_names)
        x_pd = pd.DataFrame(x)
        x_values = x_pd.values
        column_names = x_pd.columns
        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        x_values = scaler.fit_transform(x_values)
        analize_balance(y)
        x_values, y = balance_samples(x_values, y)
        train_data = train_data + list(x_values)
        labels = labels + list(y)
        partition_ranges.append( (i, i+len(y)) )
        i += len(y)

    data = pd.DataFrame(data=train_data, columns=column_names)
    data['soz'] = labels
    target = 'soz'
    predictors = column_names

    folds = [([i for i in range(len(data)) if (i < t_start or i >= t_end)],
              [i for i in range(t_start, t_end)])
             for t_start, t_end in partition_ranges]

    alg = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=6,
        scale_pos_weight=1,
        seed=7)
    
    param_test1 = {
        'n_estimators': range(100, 200, 1000),
    }
    param_test2 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }

    #grid_search(alg, param_test, folds, fit_features=data[predictors].values, to_labels=data[target].values)

    param_configs = set_param_configs(param_test_2)
    config_result = {id:{'preds':[], 'probs':[]} for id in param_configs.keys()}

    for id, c in param_configs.items():
        for train_idx, test_idx in folds:
            print('test_indexes {0}'.format(test_idx))
            train_features = data.iloc[train_idx].drop(columns=['soz']).values
            train_labels = data.iloc[train_idx]['soz']
            test_features = data.iloc[test_idx].drop(columns=['soz']).values
            test_labels = data.iloc[test_idx]['soz']
            alg.fit(train_features, train_labels, eval_metric='aucpr')
            test_predictions = alg.predict(test_features)
            test_probs = alg.predict_proba(test_features)[:, 1]
            config_result[id]['preds'] = config_result[id]['preds'] + list(test_predictions)
            config_result[id]['probs'] = config_result[id]['probs'] + list(test_probs)

    #Busco la config que tiene mejor metrica
    #Ver con Diego cual usar AP, f1score
    best_id = 1
    for id, result in config_result.items(): #probar si f1score da igual con probs y preds, ver si usa 0.5 simulando preds con ese thresh, si es eso podemos cambiar las preds segun un thresh
        average_precision = average_precision_score(labels, result['probs'])
        if average_precision > average_precision_score(labels, config_result[best_id]['probs']):
            best_id = id

def grid_search(alg, param_test, folds, fit_features, to_labels):
    gsearch = GridSearchCV(estimator=alg,
                           param_grid=param_test,
                           scoring='recall',
                           n_jobs=6,
                           iid=False,
                           cv=folds)
    gsearch.fit(fit_features, to_labels)

    print('GRID SEARCH RESULTS ')
    print(gsearch.cv_results_)
    print(gsearch.best_estimator_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)

def get_param_configs(param_test):
    for k, r in param_test.items():
        param_test[k] = [i for i in r]

    i = 0
    permutations = [[]]
    for param_values in param_test.values():
        new_permutations = []
        for v in param_values:
            for p in permutations:
                new_permutations.append(value)

    param_configs = {}
    id = 1
    for p in permutations:
        param_configs[id] = {list(param_test.keys())[i]:p[i] for i in range(len(list(param_test.keys())))}
        id += 1
    return param_configs

def set_param_config(alg, param_config):
    for feature in param_config.keys():
        alg.set_params(feature=param_config[feature])

'''
#No pude hacerla andar bien con el folds especficado, solo hace esas iteraciones e ignora el early stopping
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=4, folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          metrics='aucpr', early_stopping_rounds=early_stopping_rounds, folds=folds) #nfold=cv_folds
        print('Best fold has n_estimators: {0}'.format(cvresult.shape[0]))
        alg.set_params(n_estimators=cvresult.shape[0])
        return alg

'''
################################ PARAM TUNING END ########################################

################           ML FILTER           ##########################################

#usar para emular preds y ver si el filtro mejora el ground truth
#TODO
def f1score_goal(labels, goal):
    preds = []
    return preds

#phfo filter
#TODO fix
def phfo_filter(hfo_type_name, all_patients_dic):  # modifica e.soz,
    fpr, thresholds = compare_phfo_models(evt_type_name, loc_name, copy.deepcopy(patients_dic))
    tolerated_fpr = 0.2
    soz_confidence_thresh = get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr)
    print('Soz confidence thresh for {t} in {l} with {fp} fpr tolerance: {thresh}'.format(
        t=evt_type_name, l=loc_name, fp=tolerated_fpr, thresh=get_soz_confidence_thresh))

    model_patients_dic, without_hfo_patients = patients_with_hfos(all_patients_dic, hfo_type_name)

    only_validation_patients = False
    field_names = ml_field_names(hfo_type_name)

    if only_validation_patients:
        # Returns only a filter for validation patients
        _, target_patients = pull_apart_validation_set(model_patients_dic, hfo_type_name)
        K = 1
        test_partition = [[p.id for p in target_patients]]
        filtered_pat_dic = phfo_ml_filter(hfo_type_name, model_name, field_names, soz_confidence_thresh,
                                          model_patients_dic, target_patients, test_partition, K)
    else:
        # Returns a filter for all using cross validation
        target_patients = [p for p in model_patients_dic.values()]
        K = 3
        test_partition = get_balanced_partition(target_patients, hfo_type_name, K=K, method='deterministic')
        filtered_pat_dic = phfo_ml_filter(hfo_type_name, model_name, field_names, soz_confidence_thresh,
                                          model_patients_dic, target_patients, test_partition, K)

    for p_name, p in without_hfo_patients.items():
        filtered_pat_dic[p_name] = p

    return filtered_pat_dic


def phfo_ml_filter(hfo_type_name, model_name, field_names, soz_confidence_thresh,
                   all_patients_dic, target_patients, test_partition, K):
    target_patient_names = [p.id for p in target_patients]  # Obs mantiene el orden de target_patients
    fold = 0
    for p_names in test_partition:  # [[p.id for p in validation_patients]]
        fold += 1
        print('Running fold {f}...'.format(f=fold))
        train_patients = [p for p in target_patients if p.id not in p_names]
        test_patients = [all_patients_dic[name] for name in p_names]
        test_pat_idx = [target_patient_names.index(name) for name in p_names]
        train_features, train_labels = get_features_and_labels(train_patients, hfo_type_name, field_names)
        test_features, test_labels = get_features_and_labels(test_patients, hfo_type_name, field_names)

        train_features_pd = pd.DataFrame(train_features)
        test_features_pd = pd.DataFrame(test_features)
        # train_features_pd, test_features_pd = encode_locations(train_features_pd, test_features_pd,
        #                                                       loc_names=['loc5'])
        feature_names = test_features_pd.columns
        train_features, test_features = train_features_pd.values, test_features_pd.values

        # train_features, train_labels = balance_samples(train_features, train_labels)
        # scaler = StandardScaler()
        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        clf_preds, clf_probs = xgboost(train_features, train_labels, test_features,
                                       feature_list=feature_names, hfo_type_name=hfo_type_name)
        save_prediction(clf_preds, clf_probs, target_patients, test_pat_idx, test_labels,
                        hfo_type_name, model=model_name)

    labels, preds, probs = gather_folds(model_name, hfo_type_name, target_patients)

    i = 0
    filtered_pat_dic = dict()
    for p in target_patients:
        p_copy = copy.deepcopy(p)
        for e_copy, e in zip(p_copy.electrodes, p.electrodes):
            e_copy.hfos[hfo_type_name] = []
            for h in e.hfos[hfo_type_name]:
                if h.info['soz']: #probs[i] >= soz_confidence_thresh:
                    e_copy.add(hfo=copy.deepcopy(h))
                i += 1
        filtered_pat_dic[p.id] = p_copy

    return filtered_pat_dic


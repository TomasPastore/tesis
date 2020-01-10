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
from partition_builder import patients_with_more_than, pull_apart_validation_set, get_balanced_partition, build_folds

from metrics import print_metrics
from models import xgboost
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV


def predict_folds(folds, target_patients, hfo_type_name):
    for fold in folds:
        clf_preds, clf_probs = xgboost(fold['train_features'], fold['train_labels'], fold['test_features'],
                                       feature_list=fold['feature_names'], hfo_type_name=hfo_type_name)
        save_prediction(clf_preds, clf_probs, target_patients, fold['target_pat_idx'],
                        fold['test_labels'], hfo_type_name, model=model_name)


# Guarda los resultados del fold en la estructura global de patients
def save_prediction(clf_preds, clf_probs, target_patients, test_pat_idx, test_labels, hfo_type_name, model):
    i = 0
    for t in test_pat_idx:
        for e in target_patients[t].electrodes:
            for h in e.events[hfo_type_name]:
                assert (h.info['prediction'][model] == 0)
                h.info['prediction'][model] = int(clf_preds[i])
                if clf_probs is not None:
                    assert (h.info['proba'][model] == 0)
                    h.info['proba'][model] = clf_probs[i]
                # asserts that the hfo is being selected correctly for result i
                assert (test_labels[i] == h.info['soz'])
                i += 1


def gather_folds(model_name, hfo_type_name, target_patients):
    labels = []
    preds = []
    probs = []
    for p in target_patients:
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                labels.append(h.info['soz'])
                # Checked that classes are [False, True] order
                preds.append(h.info['prediction'][model_name])
                probs.append(h.info['proba'][model_name])

    return labels, preds, probs


def get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr):
    for i in range(len(fpr)):
        if fpr[i] == tolerated_fpr:
            return thresholds[i]
        elif fpr[i] < tolerated_fpr:
            continue
        elif fpr[i] > tolerated_fpr:
            if abs(fpr[i] - tolerated_fpr) <= abs(fpr[i - 1] - tolerated_fpr):
                return thresholds[i]
            else:
                return thresholds[i - 1]


def phfo_thresh_filter(target_patients, hfo_type_name, thresh=None, perfect=False, model_name='XGBoost'):
    if not perfect and thresh is None:
        raise ValueError('Thresh must not be None if perfect filter is off.')
    filtered_pat_dic = dict()
    for p in target_patients:
        p_copy = copy.deepcopy(p)
        for e_copy, e in zip(p_copy.electrodes, p.electrodes):
            e_copy.events[hfo_type_name] = []
            for h in e.events[hfo_type_name]:
                stay_cond = h.info['proba'][model_name] >= thresh if not perfect else h.info['soz']
                if stay_cond:
                    e_copy.add(event=copy.deepcopy(h))
        filtered_pat_dic[p.id] = p_copy

    return filtered_pat_dic


# Main function
def phfo_filter(hfo_type_name, all_patients_dic, include=None, tolerated_fpr=None, perfect=False):
    if include is None:
        include = ['model_pat', 'validation_pat']
    if not perfect and tolerated_fpr is None:
        raise ValueError('tolerated_fpr must not be None if perfect filter is off.')

    patients_dic, skipped_patients = patients_with_more_than(0, hfo_type_name, all_patients_dic)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    thresh = None
    model_name = 'XGBoost'

    if perfect:
        if 'model_pat' in include and 'validation_pat' in include:
            target_patients = [p for p in all_patients_dic.values()]
        elif 'model_pat' in include:
            target_patients=model_patients
        elif 'validation_pat' in include:
            target_patients=validation_patients
    else: #not perfect
        if 'model_pat' in include and 'validation_pat' in include:
            target_patients = [p for p in patients_dic.values()]
            model_patients = target_patients
            test_partition = get_balanced_partition(target_patients, hfo_type_name, K=4, method='deterministic')
        elif 'model_pat' in include:
            target_patients = model_patients
            test_partition = get_balanced_partition(target_patients, hfo_type_name, K=4, method='deterministic')
        elif 'validation_pat' in include:
            target_patients = validation_patients
            test_partition = [p.id for p in validation_patients]

        # Build hfo partition
        folds = build_folds(hfo_type_name, model_patients, target_patients, test_partition)

        # Predictions
        predict_folds(folds, target_patients, hfo_type_name)
        labels, preds, probs = gather_folds(model_name, hfo_type_name, target_patients=target_patients)

        # Results indicate thresh for tolerated fpr
        fpr, tpr, thresholds = roc_curve(labels, probs)
        thresh = get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr)  # solo considero phfo a los que tengan un prob de thresh o mas


    filtered_pat_dic = phfo_thresh_filter(target_patients, hfo_type_name, thresh=thresh, perfect=perfect, model_name=model_name)

    if 'model_pat' not in include or 'validation_pat' not in include:
        # We add again the patients that wouldn't be considered for the ml
        for p_name, p in skipped_patients.items():
            filtered_pat_dic[p_name] = p

    return filtered_pat_dic


################  EXPERIMENTACION ML ##############################
# TODO update
def compare_phfo_models(hfo_type_name, loc_name, patients_dic):
    print('Comparing phfo models for hfo type: {0} in {1}... '.format(hfo_type_name, loc_name))
    patients_dic, _ = patients_with_more_than(0, hfo_type_name, patients_dic)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    model_patient_names = [p.id for p in model_patients]  # Obs mantiene el orden de model_patients

    K = 3
    test_partition = get_balanced_partition(model_patients, hfo_type_name, K=K, method='deterministic')
    field_names = ml_field_names(hfo_type_name)
    fold = 0

    fig = plt.figure(0, figsize=(10, 5))
    fig.suptitle('Phfo models in {location}'.format(location=loc_name), fontsize=16)
    mean_fpr = np.linspace(0, 1, 100)
    tprs = {m: [] for m in models_to_run}
    # f_thresholds = {m: [] for m in models_to_run}
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
        # scaler = StandardScaler()
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


##################     Plotting ML results ROCS and PRE_REC          #######################
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
    plot_axe[model_name][curve_kind].plot(fpr, tpr, lw=1, alpha=0.3,
                                          label='ROC fold %d (AUC = %0.2f)' % (fold, roc_auc))

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


# PRE REC

def plot_pre_rec_fold(recall, precision, model_name, plot_axe, mean_recall, prec, aucs, fold, average_precision):
    curve_kind = 'PRE_REC'
    interp_prec = interp(mean_recall, recall, precision)
    prec[model_name].append(interp_prec)
    prec[model_name][-1][0] = 1
    auc_v = auc(recall, precision)
    aucs[model_name][curve_kind].append(auc_v)
    plot_axe[model_name][curve_kind].plot(recall, precision, lw=1, alpha=0.3,
                                          label='%s fold %d (AUC = %0.2f. AP = %0.2f.)' % (
                                              fold, auc_v, average_precision))

    return prec, aucs


def average_pre_rec(model_name, plot_axe, mean_recall, prec, aucs, aps):
    curve_kind = 'PRE_REC'
    mean_prec = np.mean(prec[model_name], axis=0)
    mean_prec[-1] = 0  # TODO revisar
    mean_auc = auc(mean_recall, mean_prec)
    std_auc = np.std(aucs[model_name][curve_kind])
    mean_ap = np.mean(aps[model_name])
    std_ap = np.std(aps[model_name])

    plot_axe[model_name][curve_kind].plot(mean_recall, mean_prec, color='b',
                                          label=r'Mean PRE_REC (AUC = %0.2f $\pm$ %0.2f. AP = %0.2f $\pm$ %0.2f)' % (
                                              mean_auc, std_auc, mean_ap, std_ap),
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


##################     Plotting ML results ROCS and PRE_REC          #######################

################################ PARAM TUNING START ########################################

def param_tuning(hfo_type_name, patients_dic):
    print('Analizying models for hfo type: {0} in {1}... '.format(hfo_type_name, 'Hippocampus'))
    patients_dic, _ = patients_with_more_than(0, hfo_type_name, patients_dic)
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
        partition_ranges.append((i, i + len(y)))
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

    # grid_search(alg, param_test, folds, fit_features=data[predictors].values, to_labels=data[target].values)

    param_configs = set_param_configs(param_test_2)
    config_result = {id: {'preds': [], 'probs': []} for id in param_configs.keys()}

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

    # Busco la config que tiene mejor metrica
    # Ver con Diego cual usar AP, f1score
    best_id = 1
    for id, result in config_result.items():  # probar si f1score da igual con probs y preds, ver si usa 0.5 simulando preds con ese thresh, si es eso podemos cambiar las preds segun un thresh
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
        param_configs[id] = {list(param_test.keys())[i]: p[i] for i in range(len(list(param_test.keys())))}
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


# usar para emular preds y ver si el filtro mejora el ground truth
# TODO
def f1score_goal(labels, goal):
    preds = []
    return preds


# TODO
def hippocampus_pathological_analisis():
    # pevent_p por electrodo en histograma
    fevent_p_by_elec = []  # porcentaje de eventos fisiologicos por electrodo
    pevent_p_by_elec = []  # porcentaje de eventos patologicos por electrodo
    phfo_counts_by_elec = []  # cantidad promedio ''
    # usar rate data para ver otras cosas como pevents_p promedio
    # TODO scatter fhfo % vs phfo % por electrodo

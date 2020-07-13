import copy
import itertools
import math as mt
import random
import graphics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from imblearn.under_sampling import CondensedNearestNeighbour, NeighbourhoodCleaningRule
from scipy import interp
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from config import models_to_run, models_dic, EVENT_TYPES
from partition_builder import patients_with_more_than, build_patient_sets, pull_apart_validation_set, \
    get_balanced_partition, build_folds

from metrics import print_metrics
from models import xgboost
from graphics import ml_training_plot
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from random import random


def ml_hfo_classifier(patients_dic, location, hfo_type,
                      use_coords, sim_recall, saving_path=None):
    print('ML CLASSIFIER location: {l} and type: {t}'.format(l=location,
                                                             t=hfo_type))

    event_type_data_by_loc = {loc_name: {}}
    loc, locations = get_locations(5, [loc_name])
    target = [
        'model_pat']  # this should be a parameter it can be ['model_pat'], ['validation_pat'],['model_pat', 'validation_pat']

    target_patients = phfo_predictor(loc_name, hfo_type_name, patients_dic,
                                     target=target, models=models_to_run,
                                     conf=sim_recall)
    if 'model_pat' in target and 'validation_pat' in target:
        name = ' baseline all'
    else:
        name = ' baseline {0}'.format(target[0])

    event_type_data_by_loc[loc_name][hfo_type_name + name] = region_info(
        {p.id: p for p in target_patients}, [hfo_type_name])

    for model_name in models_to_run:
        simulating = sim_recall is not None
        if simulating and model_name != 'Simulated':  # this is cause we need to run at least one other model to simulate, but then we just plot the simulated
            continue  # Esto ejecuta solo el simulador, esta bien, no los uso juntos

        print('Running model {0}'.format(model_name))
        labels, preds, probs = gather_folds(model_name, hfo_type_name,
                                            target_patients=target_patients)
        print('Displaying metrics for phfo classifier')
        print_metrics(model_name, hfo_type_name, labels, preds, probs)

        # SOZ HFO RATE MODEL
        fpr, tpr, thresholds = roc_curve(labels, probs)
        for tol_fpr in tol_fprs:
            thresh = get_soz_confidence_thresh(fpr, thresholds,
                                               tolerated_fpr=tol_fpr)  #
            filtered_pat_dic = phfo_thresh_filter(target_patients,
                                                  hfo_type_name,
                                                  thresh=thresh,
                                                  perfect=False,
                                                  model_name=model_name)
            confidence = None if model_name != 'Simulated' else sim_recall
            print(
                'For model_name {0} conf is {1}'.format(model_name,
                                                        confidence))
            rated_data = region_info(filtered_pat_dic, [hfo_type_name],
                                     flush=True,
                                     conf=confidence)  # calcula la info para la roc con la prob asociada al fpr tolerado
            event_type_data_by_loc[loc_name][
                hfo_type_name + ' hfo rate ' + model_name + ' FPR {0}'.format(
                    tol_fpr)] = rated_data

    graphics.event_rate_by_loc(event_type_data_by_loc,
                               metrics=['ec', 'auc'],
                               title='Hippocampal RonS HFO rate (events per minute) baseline \nVS {0}filtered rate.'.format(
                                   comp_with),
                               colors='random' if comp_with == '' else None,
                               conf=sim_recall)

def compare_Hippocampal_RonS_ml_models(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Random Forest', 'Bayes']
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=[0.6], models_to_run=models_to_run)



def predict_folds(folds, target_patients, hfo_type_name, models, conf=0.7):
    for model_name in models:
        print('Predicting folds for model {0}'.format(model_name))
        model_func = models_dic[model_name]
        for fold in folds:
            if model_name == 'Simulated':
                continue
            clf_preds, clf_probs, clf = model_func(fold['train_features'], fold['train_labels'], fold['test_features'],
                                              feature_list=fold['feature_names'], hfo_type_name=hfo_type_name)
            fold[model_name] = {}
            fold[model_name]['preds'] = clf_preds
            fold[model_name]['probs'] = clf_probs
            save_prediction(clf_preds, clf_probs, target_patients, fold['target_pat_idx'],
                            fold['test_labels'], hfo_type_name, model=model_name)

    #Generate distr for simulator
    if 'Simulated' in models:
        distr = {'FN':[], 'FP':[], 'TP':[], 'TN':[]}
        for model_name in models:
            if model_name == 'Simulated':
                continue
            labels, preds, probs = gather_folds(model_name, hfo_type_name, target_patients=target_patients)
            for i in range(len(labels)):
                l = labels[i]
                if l and preds[i]:
                    category = 'TP'
                elif l and not preds[i]:
                    category = 'FN'
                elif not l and preds[i]:
                    category = 'FP'
                elif not l and not preds[i]:
                    category = 'TN'
                distr[category].append(probs[i])
        '''
        graphics.histogram(distr['TP'], title='XGBoost predicted pHFO probability for True Positive class',
                           x_label='Predicted pHFO probability', bins=None)
        graphics.histogram(distr['FN'], title='XGBoost predicted pHFO probability for False Negative class',
                           x_label='Predicted pHFO probability', bins=None)
        graphics.histogram(distr['FP'], title='XGBoost predicted pHFO probability for False Positive class',
                           x_label='Predicted pHFO probability', bins=None)
        graphics.histogram(distr['TN'], title='XGBoost predicted pHFO probability for True Negative class',
                           x_label='Predicted pHFO probability', bins=None)
        '''
        model_name = 'Simulated'
        model_func = models_dic['Simulated']
        for fold in folds:
            clf_preds, clf_probs = model_func(fold['test_labels'], distr=distr, confidence=conf)
            fold[model_name] = {}
            fold[model_name]['preds'] = clf_preds
            fold[model_name]['probs'] = clf_probs

            save_prediction(clf_preds, clf_probs, target_patients, fold['target_pat_idx'],
                            fold['test_labels'], hfo_type_name, model=model_name)

# Guarda los resultados del fold en la estructura global de patients
def save_prediction(clf_preds, clf_probs, target_patients, test_pat_idx, test_labels, hfo_type_name, model):
    i = 0
    for t in test_pat_idx:
        for e in target_patients[t].electrodes:
            for h in e.events[hfo_type_name]:
                assert(h.info['prediction'][model] == 0)
                h.info['prediction'][model] = int(clf_preds[i])
                if clf_probs is not None:
                    assert(h.info['proba'][model] == 0)
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


def phfo_filter(hfo_type_name, patients_dic, target=None, tolerated_fpr=None, perfect=False):
    if target is None:
        target = ['model_pat', 'validation_pat']
    if not perfect and tolerated_fpr is None:
        raise ValueError('tolerated_fpr must not be None if perfect filter is off.')

    model_patients, target_patients, test_partition = build_patient_sets(target, hfo_type_name, patients_dic)
    thresh = None
    model_name = 'XGBoost'
    if not perfect:
        folds = build_folds(hfo_type_name, model_patients, target_patients, test_partition)
        predict_folds(folds, target_patients, hfo_type_name, models=[model_name])
        labels, preds, probs = gather_folds(model_name, hfo_type_name, target_patients=target_patients)

        # Results indicate thresh for tolerated fpr
        fpr, tpr, thresholds = roc_curve(labels, probs)
        thresh = get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr)

    # solo considero phfo a los que tengan un prob de thresh o mas
    filtered_pat_dic = phfo_thresh_filter(target_patients, hfo_type_name, thresh=thresh, perfect=perfect,
                                          model_name=model_name)

    ''' Esto es por si queres filtrar un par de pacientes por ej los que tienne menos de k hfos no los considero
    if 'model_pat' not in target or 'validation_pat' not in target:
        # We add again the patients that wouldn't be considered for the ml
        for p_name, p in skipped_patients.items():
            filtered_pat_dic[p_name] = p
    '''
    return filtered_pat_dic


################  EXPERIMENTACION ML ##############################
def hfo_count_quantiles(patients_dic, hfo_type_name):
    qs = [0, 0.1, 0.2, 0.25, 0.3, 0.4]
    counts = []
    names = []
    for p_name, p in patients_dic.items():
        count = 0
        for e in p.electrodes:
            count += len(e.events[hfo_type_name])

        names.append((p_name, count))
        counts.append(count)
    names.sort(key = lambda x: x[1])
    print(names)
    print('Sorted counts {0}'.format(sorted(counts)))
    quantiles = np.quantile(counts, qs, interpolation='lower')
    return {qs[i]:(quantiles[i],len([c for c in counts if c>quantiles[i]])) for i in range(len(qs))}
# devuelve un arreglo target de pacientes ya con predicciones para cada hfo en cada modelo
def phfo_predictor(loc, hfo_type_name, patients_dic, target=None, models=['XGBoost'], conf=0.7):
    if target is None:
        target = ['model_pat', 'validation_pat']
    quantiles_dic = hfo_count_quantiles(patients_dic, hfo_type_name)
    print('Hfo count quantile dic {0}'.format(quantiles_dic)) # quantiles_dic[0.3][0]
    enough_hfo_pat, excluded_pat = patients_with_more_than(quantiles_dic[0][0], hfo_type_name, patients_dic)
    model_patients, target_patients, test_partition = build_patient_sets(target, hfo_type_name, enough_hfo_pat)
    folds = build_folds(hfo_type_name, model_patients, target_patients, test_partition)
    predict_folds(folds, target_patients, hfo_type_name, models=models, conf=conf)
    plot = ml_training_plot(folds, loc, hfo_type_name, roc=True, pre_rec=True, models_to_run=models)
    for pid, p in excluded_pat.items():
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                for model_name in models:
                    h.info['prediction'][model_name]= 1
                    h.info['proba'][model_name]= 1
        target_patients.append(p)
    return target_patients



#para mejorar el baseline
#tp / tp + fn, recall alto, es decir quiero que haya pocos fn, esto implica decirm as si, mas fp, fpr  fp/N = los fhfo q permito quedarse, es desde muy estricto y buen filtro hasta baseline, permito quedar 10 porciento, entonces todos los demas son clasificados como negativos y se van. Esto no funciona porque aumenta los fn y vacia tambien los phfo, pero si no somos tann ambiciosos vamos a permitir por ej un 90% es muy baseline, despues vas para abajo

#Bayes es mas estable porque conserva mas eventos, tiene menos fn a costa de mas fp

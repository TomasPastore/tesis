import copy
from pathlib import Path
import pandas as pd
import numpy as np
import math as mt
from sklearn.metrics import roc_curve
import progressbar
import graphics
from conf import ML_MODELS_DIC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit
from partition_builder import patients_with_more_than, build_patient_sets, \
    build_folds
from soz_predictor import region_info
from utils import map_pat_ids
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn import preprocessing, metrics
from partition_builder import pull_apart_validation_set, ml_field_names
from xgboost.sklearn import XGBClassifier

#TODO review and update
def compare_Hippocampal_RonS_ml_models(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Random Forest', 'Bayes']
    #ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
    #               tol_fprs=[0.6], models_to_run=models_to_run)


# Reviewed
def compare_baseline_vs_ml(patients_dic, #Patients that built the baseline
                           plot_data_by_loc, #has baseline ROC info
                           location, #loc name
                           hfo_type, #hfo type name
                           use_coords,
                           target_patients_id='MODEL_PATIENTS',
                           ml_models=['XGBoost'],
                           tol_fprs=[0.6], #HFO filter thresh to discard
                           # fisiological hfos below the proba thresh associate
                           sim_recall=None,
                           saving_path=None):
    target_patients= ml_hfo_classifier(patients_dic, location, hfo_type,
                                       use_coords, target_patients_id,
                                       ml_models, sim_recall, saving_path)

    for model_name in ml_models:

        simulating = sim_recall is not None
        if simulating and model_name != 'Simulator':
        # this is cause we need to run at least one other mode  l to simulate,
        # but then we just plot the simulated
            continue

        #Maps predictions and probas to list in linear search of target pat
        labels, preds, probs = gather_folds(model_name, hfo_type,
                                            target_patients, estimator=np.mean)

        print('Displaying metrics for {t} in {l} ml HFO classifier using {m}'.format(t=hfo_type, l=location, m=model_name))
        print_metrics(model_name, hfo_type, labels, preds, probs)

        # SOZ HFO RATE MODEL
        fpr, tpr, thresholds = roc_curve(labels, probs)
        for tol_fpr in tol_fprs:
            thresh = get_soz_confidence_thresh(fpr, tpr, thresholds,
                                               tolerated_fpr=tol_fpr)
            filtered_pat_dic = phfo_thresh_filter(target_patients,
                                                  hfo_type,
                                                  thresh=thresh,
                                                  model_name=model_name)

            reg_loc = location if location != 'Whole Brain' else None
            loc_info = region_info(filtered_pat_dic,
                                     event_types =[hfo_type],
                                     flush =True,  # el flush es importante
                                     # porque hay que actualizar los counts
                                     conf = sim_recall,
                                     location= reg_loc,
                                     )  # calcula la info para la roc con la prob asociada al fpr tolerado
            fig_model_name = '{t}_{m}_fpr_{f}'.format(t=hfo_type,
                                                      m=model_name,
                                                      f=tol_fpr)
            plot_data_by_loc[location][fig_model_name] = loc_info

    comp_with = ''  # DONT remember why was this, i think that for colors in simu
    graphics.event_rate_by_loc(plot_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               title='HFO rate baseline VS ML pHFO filters: {t} in {l}'.format(t=hfo_type, l=location),
                               roc_saving_path = str(Path(saving_path,
                                                        location,'roc')),
                               colors='random' if comp_with == '' else None,
                               conf=sim_recall)

# Reviewed
def ml_hfo_classifier(patients_dic, location, hfo_type, use_coords,
                      target_patients_id='MODEL_PATIENTS',
                      ml_models= ['XGBoost'],
                      sim_recall=None,
                      saving_dir=None):
    '''
    Funcionalidad: Crea una lista con los target patients ya con predicciones
    tomando promedio entre testeos del boostrapping para cada modelo (ej
    XGBoost)

    Parametros:
    patients_dic: tiene como claves los patient_id y como valor un objeto
    Patient populado con lo necesario para aplicar ml con los otros parametros
    location: es la region en la cual queremos hacer ml
    hfo_type: es el tipo al cual le aplicaremos, varían los hiperparametros
    pero principalmente las features PAC.
    use_coords: indica si usar x,y,z como features en ml o no.
    target_patients_id: must be in ['MODEL_PATIENTS', 'VALIDATION_PATIENTS',
    'ALL']

    Retorna:
    # target_patients: Lista de Patients con probas de ser SOZ en patient.electrode.events[
    # hfo_type][i].info['proba']
    '''
    print('ML HFO classifier for location: '
          '{l} and type: {t}'.format(l=location, t=hfo_type))
    print('target_patients_id: {t}'.format(t=target_patients_id))
    print('saving_dir: {0}'.format(saving_dir))
    # This is if we want to exclude from ml analysis the patients that have
    # less than N events so their electrode rates will remain unchanged
    quantiles_dic = hfo_count_quantiles(patients_dic, hfo_type)
    print('Hfo count quantile dic {0}'.format(quantiles_dic))  # ej
    # quantiles_dic[0.3][0]
    enough_hfo_pat, excluded_pat = patients_with_more_than(quantiles_dic[0][0],
                                                           patients_dic,
                                                           hfo_type
                                                           )

    model_patients, target_patients, test_partition = build_patient_sets(
        target_patients_id, enough_hfo_pat, location)

    folds = build_folds(hfo_type, model_patients,
                        target_patients, test_partition)

    predict_folds(folds, target_patients, hfo_type,
                  models=ml_models, sim_recall=sim_recall)

    plot = graphics.ml_training_plot(target_patients, location, hfo_type, roc=True,
                                     pre_rec=True, saving_dir=saving_dir)

    # Esto incluye a los excluidos por cuantil con todos sus eventos sin filtro
    for pat_id, p in excluded_pat.items():
        for e in p.electrodes:
            for h in e.events[hfo_type]:
                for model_name in ml_models:
                    h.info['prediction'][model_name] = 1
                    h.info['proba'][model_name] = 1
        target_patients.append(p)
    return target_patients


# Reviewed
# Saves the results of each model in target patients info dic
def predict_folds(folds, target_patients, hfo_type_name, models,
                  sim_recall=0.7):
    for model_name in [m for m in models if m != 'Simulator']:
        print('Predicting folds for model {0}'.format(model_name))
        bar = progressbar.ProgressBar(maxval=len(folds),
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ',
                                               progressbar.Percentage()])
        bar.start()
        model_func = ML_MODELS_DIC[model_name]
        for fold_i, fold in enumerate(folds):
            fold_i += 1 #enumerate starts at 0 but I want update bar from 1
            clf_preds, clf_probs, clf = model_func(fold['train_features'],
                                                   fold['train_labels'],
                                                   fold['test_features'],
                                                   feature_list=fold[
                                                       'feature_names'],
                                                   hfo_type_name=hfo_type_name)
            fold[model_name] = {}
            fold[model_name]['preds'] = clf_preds
            fold[model_name]['probs'] = clf_probs
            save_prediction(clf_preds, clf_probs, target_patients,
                            fold['target_pat_idx'],
                            fold['test_labels'], hfo_type_name,
                            model=model_name)
            bar.update(fold_i)
        bar.finish()
    #print(folds)
    # Generate distr for simulator
    if 'Simulator' in models:
        distr = {'FN': [], 'FP': [], 'TP': [], 'TN': []}
        # Estimo distribuciones de probas para el simulador a partir de otros
        # clasificadores
        for model_name in models:
            if model_name == 'Simulator':
                continue
            labels, preds, probs = gather_folds(model_name, hfo_type_name,
                                                target_patients=target_patients)
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
        # Histogramas de probabilidad para las clases TP, TN, FN, FP  
        class_name = 'TP'
        graphics.histogram(distr[class_name], title=class_name+' proba distr',
                           x_label='SOZ Probability', bins=None)
        '''
        model_name = 'Simulator'
        print('Predicting folds for model {0}'.format(model_name))
        bar = progressbar.ProgressBar(maxval=len(folds),
                                      widgets=[progressbar.Bar('=', '[', ']'),
                                               ' ',
                                               progressbar.Percentage()])
        bar.start()
        simulator_func = ML_MODELS_DIC['Simulated']
        for fold_i, fold in enumerate(folds):
            fold_i += 1
            clf_preds, clf_probs = simulator_func(fold['test_labels'],
                                                distr=distr,
                                              confidence=sim_recall)
            fold[model_name] = {}
            fold[model_name]['preds'] = clf_preds
            fold[model_name]['probs'] = clf_probs

            save_prediction(clf_preds, clf_probs, target_patients,
                            fold['target_pat_idx'],
                            fold['test_labels'], hfo_type_name,
                            model=model_name)
            bar.update(fold_i)
        bar.finish()
# Reviewed
# Guarda los resultados del fold en la estructura global de patients
# Es importante que el orden de pacientes los indices de test_pat_idx fue el
# mismo que el de las test_labels como precondicion, que lo hace build_folds
def save_prediction(clf_preds, clf_probs, target_patients, test_pat_idx,
                    test_labels, hfo_type_name, model):
    i = 0
    for t in test_pat_idx:
        for e in target_patients[t].electrodes:
            for h in e.events[hfo_type_name]:
                h.info['prediction'][model].append(int(clf_preds[i]))
                h.info['proba'][model].append(clf_probs[i])
                # asserts that the hfo is being selected correctly for result i
                assert (test_labels[i] == h.info['soz'])
                i += 1

# Reviewed
def gather_folds(model_name, hfo_type_name, target_patients, estimator=np.mean):
    labels = []
    preds = []
    probs = []
    for p in target_patients:
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                labels.append(e.soz)
                # Checked that classes are [False, True] order
                prediction = 1 if h.info['prediction'][model_name].count(1) \
                                  >= h.info['prediction'][model_name].count(
                    0) else 0
                preds.append(prediction)
                probs.append(estimator(h.info['proba'][model_name]))
    return labels, preds, probs

# Reviewed
def get_soz_confidence_thresh(fpr, tpr, thresholds, tolerated_fpr):
    def print_thresh_info(tol_fpr, i):
        print('For tolerated fpr {t_fpr} --> Proba Thresh: {p_thresh}, '
              'TPR: {tpr}'.format(t_fpr=tol_fpr, p_thresh=
        thresholds[i], tpr=tpr[i]))
    for i in range(len(fpr)):
        if fpr[i] == tolerated_fpr:
            print_thresh_info(tolerated_fpr, i)
            return thresholds[i]
        elif fpr[i] < tolerated_fpr:
            continue
        elif fpr[i] > tolerated_fpr:
            if abs(fpr[i] - tolerated_fpr) <= abs(fpr[i - 1] - tolerated_fpr):
                print_thresh_info(tolerated_fpr, i)
                return thresholds[i]
            else:
                print_thresh_info(tolerated_fpr, i-1)
                return thresholds[i - 1]

# Reviewed
# Filters the Events whose predicted probability of being SOZ is greater than
# the threshold given by parameter
def phfo_thresh_filter(target_patients, hfo_type_name, thresh=None, model_name='XGBoost'):
    filtered_pat_dic = dict()
    for p in target_patients:
        p_copy = copy.deepcopy(p)
        for e_copy, e in zip(p_copy.electrodes, p.electrodes):
            e_copy.events[hfo_type_name] = []
            for h in e.events[hfo_type_name]:
                if h.info['proba'][model_name] >= thresh:
                    e_copy.add(event=copy.deepcopy(h))
            e_copy.flush_cache([hfo_type_name]) #recalculates event counts
        filtered_pat_dic[p.id] = p_copy

    return filtered_pat_dic

#TODO
def phfo_filter(hfo_type_name, patients_dic, target=None, tolerated_fpr=None,
                ):
    if target is None:
        target = ['model_pat', 'validation_pat'] #acordate q ahroa es un string

    model_patients, target_patients, test_partition = build_patient_sets(target,
                                                                         hfo_type_name,
                                                                         patients_dic)
    thresh = None
    model_name = 'XGBoost'
    perfect = False
    if not perfect:
        folds = build_folds(hfo_type_name, model_patients, target_patients,
                            test_partition)
        predict_folds(folds, target_patients, hfo_type_name,
                      models=[model_name])
        labels, preds, probs = gather_folds(model_name, hfo_type_name,
                                            target_patients=target_patients)

        # Results indicate thresh for tolerated fpr
        fpr, tpr, thresholds = roc_curve(labels, probs)
        thresh = get_soz_confidence_thresh(fpr, tpr, thresholds, tolerated_fpr)

    # solo considero phfo a los que tengan un prob de thresh o mas
    filtered_pat_dic = phfo_thresh_filter(target_patients, hfo_type_name,
                                          thresh=thresh,
                                          model_name=model_name)
    '''
    if 'model_pat' not in target or 'validation_pat' not in target:
        # We add again the patients that wouldn't be considered for the ml
        for p_name, p in skipped_patients.items():
            filtered_pat_dic[p_name] = p
    '''
    return filtered_pat_dic


# Reviewed
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
    names.sort(key=lambda x: x[1])
    print(names)
    print('Sorted Patient HFO counts {0}'.format(sorted(counts)))
    quantiles = np.quantile(counts, qs, interpolation='lower')
    # devuelve key:cuantil; value: cuantos eventos necesito tener al
    # menos para ser superior al cuantil de la key, cuantos pacientes quedan
    # despues del filtro
    return {qs[i]: (quantiles[i], len([c for c in counts if c > quantiles[i]]))
            for i in range(len(qs))}

def print_metrics(model, hfo_type_name, y_test, y_pred, y_probs):
    print('')
    print('-------------------------------------------')
    print('Displaying metrics for {0} using {1} model:'.format(hfo_type_name, model))
    #print('ROC AUC of ---> {0}'.format(roc_auc_score(y_test, y_probs)))
    print('Confusion matrix')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
   # average_precision = average_precision_score(y_test, y_probs)
    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('-------------------------------------------')
    print('')




def youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]

#Returns best estimator params for validation in location
def ml_hfo_classifier_sk_learn_train(patients_dic,
                                     location,
                                     hfo_type,
                                     use_coords,
                                     ml_models= ['XGBoost'],
                                     sim_recall=None,
                                     saving_dir=None):
    '''
    Parametros:
    patients_dic: tiene como claves los patient_id y como valor un objeto
    Patient populado con lo necesario para aplicar ml en location siguiendo los
    otros parametros que tambien se indican
    location: es la region en la cual se hace ml
    hfo_type: es el tipo al cual le aplicaremos, varían los hiperparametros
    pero principalmente las features PAC.
    use_coords: indica si usar x,y,z como features en ml o no.
    '''
    print('ML HFO classifier for location: '
          '{l} and type: {t}'.format(l=location, t=hfo_type))
    print('saving_dir: {0}'.format(saving_dir))
    model_patients, validation_patients = pull_apart_validation_set(
        patients_dic, location, val_size=0.3)
    field_names = ml_field_names(hfo_type, include_coords=use_coords)
    X, y, groups = serialize_patients_to_events(model_patients, hfo_type, field_names)
    cv = GroupShuffleSplit(n_splits=1000, test_size=.3, random_state=42)

    #For model in models
    xgboost = XGBClassifier(learning_rate=0.05,
                        n_estimators=100,  # 100
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
                        eval_metric='aucpr'  # 'aucpr'
                        )

    clf = make_pipeline(preprocessing.StandardScaler(), xgboost)
    def tn(y, y_pred): return metrics.confusion_matrix(y, y_pred)[0, 0]
    def fp(y, y_pred): return confusion_matrix(y, y_pred)[0, 1]
    def fn(y, y_pred): return confusion_matrix(y, y_pred)[1, 0]
    def tp(y, y_pred): return confusion_matrix(y, y_pred)[1, 1]
    scoring = {'precision': 'precision',
               'recall': 'recall',
               'tp': metrics.make_scorer(tp),
               'tn': metrics.make_scorer(tn),
               'fp': metrics.make_scorer(fp),
               'fn': metrics.make_scorer(fn),
               'balanced_accuracy': metrics.make_scorer(
                   metrics.balanced_accuracy,
                   adjusted=True),
               'f1_score': 'f1',
               'average_precision': 'average_precision',
               'roc_auc': 'roc_auc',
               }
    cv_results = cross_validate(clf, X, y, groups, scoring, cv=cv, n_jobs=-1,
                                return_estimator=True)
    print(cv_results)

    '''
    probas = cross_val_predict(clf, X, y, groups, cv=cv, n_jobs=-1,
                               method= 'predict_proba')
    data_by_model['XGBoost'] = dict()
    data_by_model['XGBoost']['probas'] = probas
    data_by_model['XGBoost']['y'] = y
    plot = graphics.ml_training_plot(data_by_model, location, hfo_type,
                                     roc=True, pre_rec=True,
                                     saving_dir=saving_dir)
    '''

    '''
    # TODO manual
    for train_idx, test_idx in gss_validation.split(X, y, groups):
        print("TRAIN:", train_idx, "TEST:", test_idx)
    '''




def serialize_patients_to_events(model_patients, hfo_type, field_names):
    X, y, groups = [], [], []
    pac = [f for f in field_names if 'angle' in f or 'vs' in f]
    groups = map_pat_ids(model_patients)
    for p in model_patients:
        for e in p.electrodes:
            for h in e.events[hfo_type]:
                if all([isinstance(h.info[f], float) for f in
                        pac]):  # I use
                    # this event only if all the pac is not null, else skip,
                    # if you don't use any '_angle' or 'vs' PAC property this takes
                    # every event
                    feature_row_i = {}
                    for feature_name in field_names:
                        if 'angle' in feature_name or 'vs' in feature_name:
                            feature_row_i[
                                'SIN({0})'.format(feature_name)] = mt.sin(
                                h.info[feature_name])
                            feature_row_i[
                                'COS({0})'.format(feature_name)] = mt.cos(
                                h.info[feature_name])
                        else:
                            feature_row_i[feature_name] = h.info[
                                feature_name]
                    X.append(feature_row_i)
                    y.append(h.info['soz'])

    X_pd = pd.DataFrame(X)
    feature_names = X_pd.columns  # adds sin and cos for PAC
    return X_pd.values, np.array(y), np.array(groups)
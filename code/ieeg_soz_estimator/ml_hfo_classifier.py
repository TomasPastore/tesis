import copy
from pathlib import Path
import pandas as pd
import numpy as np
import math as mt
from sklearn.metrics import roc_curve
import progressbar
import graphics
from config import models_dic
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
from random import choices
#TODO
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
        model_func = models_dic[model_name]
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
        simulator_func = models_dic['Simulator']
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

'''

# REMOVER MODULO, METER
def param_tuning(hfo_type_name, patients_dic):
    print('Analizying models for hfo type: {0} in {1}... '.format(hfo_type_name, 'Hippocampus'))
    patients_dic, _ = patients_with_more_than(0, patients_dic, hfo_type_name)
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


# Como se guardan iterando patients_dic.values despues hay que
# iterarlo igual y crear un diccionario copia sin alterar el viejo



# TODO flush evt count after filter
# predictors = ['freq_av', 'duration', 'power_pk']
# data = {predictor: [] for predictor in predictors}
#for pred in predictors:
#    data[pred].append(evt.info[pred])
# Note: Maybe using meadian would be better since the mean gets more
        # affected
        # by outliers
# graphics.k_means_clusters_plot(data)


def artifact_filter(hfo_type, patients_dic):
    '''
    Filters Fast RonO near 300 HZ and RonO near 180 Hz electrical artifacts.
    :param hfo_type: The hfo type with electrical artifacts
    :param patients_dic: Patient, Electrode, Event data structures
    :return: modified patients dic for type hfo_type
    '''
    print('Entering filter for electrical artifacts')
    remove_from_elec_by_pat = {p_name: [] for p_name in patients_dic.keys()}
    # For each patient I keep a list of elec names where we can gradually
    # remove candidates and update
    if hfo_type == 'Fast RonO':
        artifact_freq = 300 # HZ
        art_radius = 20 # hz
        pw_line_int = 60 # HZ
        artifact_cnts = dict() # 300 HZ +- art_radius event counts for each patient
        physio_cnts = [] # 360 HZ +- art_radius event counts for each patient
        for p_name, p in patients_dic.items():
            artifact_cnt = 0
            physio_cnt = 0
            for e in p.electrodes:
                for evt in e.events['Fast RonO']:
                    if (artifact_freq - art_radius) <= evt.info['freq_av'] and \
                            evt.info['freq_av'] <= (artifact_freq + art_radius):
                        artifact_cnt += 1
                        remove_from_elec_by_pat[p_name].append(e.name)

                    elif (artifact_freq + pw_line_int - art_radius) <= evt.info['freq_av'] and \
                            evt.info['freq_av'] <= (artifact_freq + pw_line_int + art_radius):
                        physio_cnt += 1
            artifact_cnts[p_name] = artifact_cnt
            physio_cnts.append(physio_cnt)

        # Saving stats
        artifact_mean = sum(list(artifact_cnts.values())) / (len(
            artifact_cnts.keys())-1)
        artifact_std = np.std(list(artifact_cnts.values()), ddof=1)
        physio_mean = sum(physio_cnts) / (len(physio_cnts)-1)
        physio_std = np.std(physio_cnts, ddof=1)
        print('-----------------------------------')
        print('\nFRonO Artifacts (300 HZ +- {0})'.format(art_radius))
        print('Sample artifact mean', artifact_mean)
        print('Sample artifact std', artifact_std)
        print('\nFRonO Phisiological (360 HZ +- {0})'.format(art_radius))
        print('Sample physiological mean', physio_mean)
        print('Sample phisiological std', physio_std)
        print('-----------------------------------')
        # Removing artifacts
        for p_name, p in patients_dic.items():
            remove_cnt = max(0, int(artifact_cnts[p_name] - physio_mean) )
            print('For patient {0} we remove {1} events'.format(p_name,
                                                                remove_cnt))
            for i in range(remove_cnt):
                elec_to_rmv = choices(remove_from_elec_by_pat[p_name], k=1)[0]
                remove_from_elec_by_pat[p_name].remove(elec_to_rmv)
                electrode = p.get_electrode(elec_to_rmv)
                electrode.remove_rand_evt(hfo_type='Fast RonO', art_radius=art_radius)

            for e in p.electrodes:
                e.flush_cache(['Fast RonO']) #Recalc events counts for hfo rate

    else:
        print('Not implemented filter type')
        raise NotImplementedError()

    return patients_dic


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
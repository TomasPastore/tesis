import copy

import numpy as np
from sklearn.metrics import roc_curve

import graphics
from config import models_dic
from graphics import ml_training_plot
from metrics import print_metrics
from partition_builder import patients_with_more_than, build_patient_sets, \
    build_folds

#TODO
def compare_Hippocampal_RonS_ml_models(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Random Forest', 'Bayes']
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=[0.6], models_to_run=models_to_run)


# Reviewed
def compare_baseline_vs_ml(patients_dic, #Patients that built the baseline
                           plot_data_by_loc, #has baseline ROC info
                           location, #loc name
                           hfo_type, #hfo type name
                           use_coords,
                           target_patients_id=['model_patients'],
                           ml_models=['XGBoost'],
                           tol_fprs=[0.6], #HFO filter thresh to discard
                           # fisiological hfos below the proba thresh associate
                           sim_recall=None,
                           saving_path=None):
    target_patients= ml_hfo_classifier(patients_dic, location, hfo_type,
                                       use_coords, target_patients_id,
                                       ml_models, sim_recall, saving_path)

    event_type_data_by_loc = {location: {}}
    for model_name in ml_models:

        simulating = sim_recall is not None
        if simulating and model_name != 'Simulator':
        # this is cause we need to run at least one other mode  l to simulate,
        # but then we just plot the simulated
            continue

        #Maps predictions and probas to list in linear search of target pat
        labels, preds, probs = gather_folds(model_name, hfo_type,
                                            target_patients, estimator=np.mean)

        print('Displaying metrics for {t} in {l} ml HFO classifier using {'
              'm}'.format(t=hfo_type, l=location, m=model_name))
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

            reg_loc = loc_name if loc_name != 'Whole Brain' else None
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
    graphics.event_rate_by_loc(event_type_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               roc_saving_path=str(Path(saving_path,
                                                        'loc_{g}'.format(
                                                            g=granularity),
                                                        '3_iii_sleep_tagged')),
                               change_tab_path=True)

    graphics.event_rate_by_loc(plot_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               title='HFO rate baseline VS ML pHFO filters: {'
                                     't} in {l}'.format(t=hfo_type, l=location),
                               roc_saving_path = str(Path(saving_path,
                               'loc_{g}'.format(
                                   g=granularity),
                               '3_iii_sleep_tagged')),
                               colors='random' if comp_with == '' else None,
                               conf=sim_recall)

# Reviewed
def ml_hfo_classifier(patients_dic, location, hfo_type, use_coords,
                      target_patients_id=['model_patients'],
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
        target_patients_id, hfo_type, enough_hfo_pat)

    folds = build_folds(hfo_type, model_patients,
                        target_patients, test_partition)

    predict_folds(folds, target_patients, hfo_type,
                  models=ml_models, sim_recall=sim_recall)

    plot = ml_training_plot(folds, location, hfo_type, roc=True,
                            pre_rec=True, models_to_run=ml_models,
                            saving_dir=saving_dir)

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
    for model_name in models:
        print('Predicting folds for model {0}'.format(model_name))
        model_func = models_dic[model_name]
        for fold in folds:
            if model_name == 'Simulator':
                continue # Necesito primero los otros para estimar el simulador
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
        simulator_func = models_dic['Simulator']
        for fold in folds:
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
                labels.append(h.info['soz'])
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
        target = ['model_pat', 'validation_pat']

    model_patients, target_patients, test_partition = build_patient_sets(target,
                                                                         hfo_type_name,
                                                                         patients_dic)
    thresh = None
    model_name = 'XGBoost'
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
                                          thresh=thresh, perfect=perfect,
                                          model_name=model_name)

    if 'model_pat' not in target or 'validation_pat' not in target:
        # We add again the patients that wouldn't be considered for the ml
        for p_name, p in skipped_patients.items():
            filtered_pat_dic[p_name] = p
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


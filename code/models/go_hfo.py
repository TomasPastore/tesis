import math as mt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymongo
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from classes import Database, Patient, Electrode, HFO
from config import HFO_TYPES, models_to_run, models_to_run_obj, type_names_to_run
from metrics import print_metrics
from models import hfo_rate
from preprocessing import soz_bool, parse_electrodes, parse_hfos, encode_type_name
from utils import log


def load_patients(hfo_collection, electrodes_collection, hfo_type_names):
    # Todo 'outcome', 'resected',
    common_attr = ['patient_id', 'age', 'file_block', 'electrode', 'loc5', 'soz', 'soz_sc',
                   'type', 'duration', 'fr_duration', 'r_duration',
                   'freq_av', 'freq_pk', 'power_av', 'power_pk']

    patients_by_hfo_type = {hfo_type_name:dict() for hfo_type_name in HFO_TYPES}
    for hfo_type_name in hfo_type_names:

        add_patients_by_electrodes(patients_by_hfo_type[hfo_type_name], electrodes_collection)
        add_patients_by_hfos(
            patients_by_hfo_type[hfo_type_name],
            hfo_collection,
            common_attr,
            hfo_type_name
        )
        add_empty_blocks(patients_by_hfo_type[hfo_type_name], electrodes_collection)

    return patients_by_hfo_type

def add_patients_by_electrodes(patients_of_hfo_type, electrodes_collection):
    elec_cursor = electrodes_collection.find(
        filter = {'loc5': 'Hippocampus'},
        projection = ['patient_id', 'age', 'file_block', 'electrode', 'loc5', 'soz', 'soz_sc']
    )
    parse_electrodes(patients_of_hfo_type, elec_cursor)

def add_patients_by_hfos(patients_of_hfo_type, hfo_collection, common_attr, hfo_type_name):

    if hfo_type_name in ['RonO', 'Fast RonO']:
        specific_attributes = ['slow', 'slow_vs', 'slow_angle',
                               'delta', 'delta_vs', 'delta_angle',
                               'theta', 'theta_vs', 'theta_angle',
                               'spindle', 'spindle_vs', 'spindle_angle']
    else:
        specific_attributes = ['spike', 'spike_vs', 'spike_angle']

    hfo_type = encode_type_name(hfo_type_name)
    hfos_cursor = hfo_collection.find(
        filter={'type': hfo_type, 'loc5': 'Hippocampus', 'intraop':'0'},
        projection=common_attr + specific_attributes,
        #sort= [('patient_id', pymongo.ASCENDING), ('electrode', pymongo.ASCENDING)]
    )
    #Unifying types and parsing inconsistencies
    parse_hfos(patients_of_hfo_type, hfos_cursor)

def add_empty_blocks(patients, electrodes_collection):
    # Nota: esto agrega los bloques sin hfos para los (patient,electrode) que tienen otro bloque de ese electrodo
    # con hfos en hipocampo, no considera los patient electrodes con loc5 en hipocampo que no tengan ningun hfo
    empty_blocks_added = 0
    for p in patients.values():
        for e in p.electrodes:
            hfo_empty_blocks = electrodes_collection.find(
                filter={'patient_id': p.id, "$or": [{'electrode': [e.name]}, {'electrode': e.name}]},
                projection=['soz', 'file_block']
            )
            for electrode_rec in hfo_empty_blocks:
                # Consistency for soz
                soz = soz_bool(electrode_rec['soz'])
                if (soz != e.soz):
                    log(msg=('Warning, soz disagreement among hfos in '
                             'the same patient_id, electrode, '
                             'running OR between values'),
                        msg_type='SOZ_0',
                        patient=p.id,
                        electrode=e.name
                        )
                    e.soz = e.soz or soz

                    # Add block id for hfo_rate
                    file_block = int(electrode_rec['file_block'])
                    if file_block not in p.file_blocks.keys():
                        empty_blocks_added += 1
                        p.file_blocks[file_block] = None

    print('Empty blocks added: {0}'.format(empty_blocks_added))


def get_feature_names(hfo_type_name):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        feature_names = ['duration', 'freq_pk', 'power_pk',
                        'slow', 'slow_vs', 'slow_angle',
                        'delta', 'delta_vs', 'delta_angle',
                         'theta', 'theta_vs', 'theta_angle',
                         'spindle', 'spindle_vs', 'spindle_angle']
    else:
        feature_names = ['duration', 'freq_pk', 'power_pk',
                         'spike', 'spike_vs', 'spike_angle']

    return feature_names

def split_patients_2(patients, train_p=0.6):
    assert (train_p <= 1 and train_p >= 0)
    patient_count = len(patients)
    train_size = int(patient_count * train_p)
    test_size = int(patient_count * (1 - train_p))
    train_size += patient_count - (train_size + test_size)
    train_set = patients[:train_size]
    test_set = patients[train_size:patient_count]

    return train_set, test_set

def split_patients_3(patients, train_p=0.6, test_p=0.2, val_p=0.2):
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
                labels.append(h.info['soz']) #Using the soz of the hfo, not the electrode

    return features, labels

#Patients are restricted to one type of hfo
def get_partition(patients, train_idx, test_idx, hfo_type_name, feature_names):

    train_patients = list(np.array(patients)[train_idx])
    test_patients = list(np.array(patients)[test_idx])

    train_features, train_labels = get_features_and_labels(train_patients, hfo_type_name, feature_names)

    test_features, test_labels = get_features_and_labels(test_patients, hfo_type_name, feature_names)

    train_features = pd.DataFrame(train_features).values
    train_labels = np.array(train_labels)
    #train_labels = pd.DataFrame(train_labels).values

    test_features = pd.DataFrame(test_features).values
    test_labels = np.array(test_labels)
    #test_labels = pd.DataFrame(test_labels).values

    return train_features, train_labels, test_features, test_labels


def contribute_to_metrics(clf_preds, clf_probs, patients, test_idx, test_labels, hfo_type_name, K, model):
    i = 0
    for t in test_idx:
        for e in patients[t].electrodes:
            for h in e.hfos[hfo_type_name]:
                try:
                    h.info['prediction'][model] = [h.info['prediction'][model][0] + int(not bool(clf_preds[i])) ,
                                                   h.info['prediction'][model][1] + int(clf_preds[i])]
                    h.info['proba'][model] += clf_probs[i] / K
                    assert(h.info['soz'] == bool(test_labels[i]))
                except TypeError as e:
                    print('Patient name {0}'.format(patients[t].id))
                    print(h.info['prediction'])
                i += 1

def reset_metrics(patients, hfo_type_name):
    for p in patients:
        for e in p.electrodes:
            for h in e.hfos[hfo_type_name]:
                h.reset_preds()

def map_hfos(patients, hfo_type_name):
    hfos = []
    pat_ranges = []
    j = 0
    k = 0
    for p in patients:
        for e in p.electrodes:
            for h in e.hfos[hfo_type_name]:
                hfos.append(h)
                k+=1
        pat_ranges.append((j,k))
        j=k
    return hfos, pat_ranges

def get_feature(i,j, features, name):
    if 'SIN' in name:
        return name[4:-1],mt.asin(features[i][j]) #removes 'SIN()'
    elif 'COS' in name:
        return name[4:-1],mt.acos(features[i][j]) #removes 'COS()'
    else:
        return name,features[i][j]

def main():
    db = Database()
    connection = db.get_connection()
    db = connection.deckard_new
    #Ver pipeline scyikit learn
    electrodes_collection = db.Electrodes
    electrodes_collection.create_index([('type', pymongo.ASCENDING)], unique = False)
    electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

    hfo_collection = db.HFOs
    hfo_collection.create_index([('type', pymongo.ASCENDING)], unique = False)
    hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

    '''
    print('Unique patients from ELECTRODES COLLECTION with electrodes in Hippocampus:')
    with_electrodes_in_hip = unique_patients(electrodes_collection, {'loc5': 'Hippocampus'})
    print('Unique patients from HFOs COLLECTION with electrodes in Hippocampus and intraop==0')
    with_hfo_in_hip_and_inop_0 = unique_patients(hfo_collection, {'intraop':'0', 'loc5': 'Hippocampus'})
    '''

    print('HFO types to run: {0}'.format(type_names_to_run))
    print('Loading data from database...')
    patients_by_hfo_type = load_patients(hfo_collection, electrodes_collection, type_names_to_run)

    #if DEBUG:
        #print('Inconsistencies found while parsing: {0}'.format(inconsistencies))

    for hfo_type_name in type_names_to_run:
        patients_dic = patients_by_hfo_type[hfo_type_name]
        patients = [p for p in patients_dic.values()]
        hfos, pat_ranges = map_hfos(patients, hfo_type_name) #pat_ranges[i]= (j,k) means the i patients hfos are hfos[j:k]
        mean_hfo_per_pat = np.mean([k-j for j,k in pat_ranges])
        print('Mean hfo count per patient for {0} is {1}'.format(hfo_type_name, mean_hfo_per_pat))
        feature_names = get_feature_names(hfo_type_name)
        features, labels = get_features_and_labels(patients, hfo_type_name, feature_names)
        features_pd = pd.DataFrame(features)
        features = features_pd.values
        features_cols = features_pd.columns
        labels=np.array(labels)
        prev_count =  len(features)
        '''
        print('Performing resample with SMOTETomek...')
        smt = SMOTETomek(random_state=42)
        features, labels = smt.fit_resample(features, labels)
        post_count =  len(features)
        dif = post_count - prev_count
        print('Added {0} instances to balance classes...'.format(dif))
        i = prev_count
        j = prev_count
        k = prev_count
        while(dif>0):
            patient = Patient(id='Interpolated_{0}'.format(j))
            electrode = Electrode(
                'emulated_electrode',
                bool(labels[i]),
                soz_sc=bool(labels[i]),
                loc5='Hippocampus'
            )
            patient.add_electrode(electrode)
            while(dif > 0 and k < j+mean_hfo_per_pat):
                hfo = HFO(info = dict([get_feature(i,j,features, name)  for j, name in enumerate(features_cols)]))
                hfo.info['type'] = hfo_type_name
                hfo.info['soz']= labels[i]
                hfo.info['prediction'] = {m: [0, 0] for m in models_to_run}
                hfo.info['proba'] = {m: 0 for m in models_to_run}
                electrode.add(hfo)
                hfos.append(hfo)
                i+=1
                k+=1
                dif-=1

            pat_ranges.append((j,k))
            patients.append(patient)
            j=k
    '''
        # Baseline
        hfo_rate(patients, hfo_type_name)

        #Machine learning
        mean_fpr = np.linspace(0, 1, 100)
        K = 2
        kf = KFold(n_splits=K)

        i=0 #iteration
        tprs = {m:[] for m in models_to_run}
        aucs = {m:[] for m in models_to_run}
        plt.figure()
        bayes_axe = plt.subplot(231)
        rf_axe = plt.subplot(232)
        svm_axe = plt.subplot(233)
        balanced_rf_axe = plt.subplot(234)
        xgboost_axe= plt.subplot(236)

        plot_axe = {
            'Naive Bayes': bayes_axe,
            'Random Forest': rf_axe,
            'SVM': svm_axe,
            'Balanced RF' : balanced_rf_axe,
            'XGBoost': xgboost_axe
        }

        for train_idx, test_idx in kf.split( [i for i in range(len(pat_ranges))] ):

            train_features, train_labels, test_features, test_labels = get_partition(
                patients,
                train_idx,
                test_idx,
                hfo_type_name,
                feature_names,
            )

            #TODO: XGBOOST && Sagemaker
            # Scaling
            sc = StandardScaler()
            train_features = sc.fit_transform(train_features)
            test_features = sc.transform(test_features)
            #Models
            for model_name, model_func in zip(models_to_run, models_to_run_obj):

                clf_preds, clf_probs = model_func(train_features, train_labels, test_features)
                contribute_to_metrics(clf_preds, clf_probs, patients, test_idx, test_labels, hfo_type_name, K, model=model_name)

                ##ROCs!
                fpr, tpr, thresholds = roc_curve(test_labels, clf_probs)
                tprs[model_name].append(interp(mean_fpr, fpr, tpr))
                tprs[model_name][-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs[model_name].append(roc_auc)
                plot_axe[model_name].plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i += 1

        ######## AVERAGE RESULTS AMONG FOLDS ###########
        for model_name in models_to_run:
            test_labels = []
            preds = []
            probs = []
            for p in patients:
                for e in p.electrodes:
                    for h in e.hfos[hfo_type_name]:
                        test_labels.append(h.info['soz'])
                        preds.append( 1 if h.info['prediction'][model_name][1] >= h.info['prediction'][model_name][0] else 0 ) #Checked that classes are [False, True] order
                        probs.append(h.info['proba'][model_name])

            print_metrics(test_labels, preds, probs, hfo_type_name, model_name)


            ##ROCs
            plot_axe[model_name].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs[model_name], axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs[model_name])
            plot_axe[model_name].plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                     lw=2, alpha=.8)

            std_tpr = np.std(tprs[model_name], axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plot_axe[model_name].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                             label=r'$\pm$ 1 std. dev.')

            plot_axe[model_name].set_xlim([-0.05, 1.05])
            plot_axe[model_name].set_ylim([-0.05, 1.05])
            plot_axe[model_name].set_xlabel('False Positive Rate')
            plot_axe[model_name].set_ylabel('True Positive Rate')
            plot_axe[model_name].set_title('ROC curve for {0} using {1} model'.format(hfo_type_name, model_name))
            plot_axe[model_name].legend(loc="lower right")
        plt.savefig('/home/tpastore/{0}_ROC_comparison.png'.format(hfo_type_name), format='png')
        plt.show()

            ##END ROCs

if __name__ == "__main__":
    main()
from sys import version as py_version
import warnings
from pathlib import Path

from patient import Patient

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from config import (EVENT_TYPES, HFO_TYPES, exp_save_path,
                    experiment_default_path,
                    models_to_run)
from db_parsing import get_granularity, all_loc_names, \
    preference_locs, load_patients

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
from utils import print_info, time_counter
import graphics


# Maybe this is not necesary having the localized verison review.
# 3) Predicting SOZ with rates: Baselines  #####################################
# 3i
# Make the ROC baselines for whole brain.
# Returns the data for the ml and the data to plot the baselines
# TODO pAUC
def evt_rate_soz_pred_baseline_whole_brain(elec_collection, evt_collection,
                                           intraop=False,
                                           load_untagged_coords_from_db=True,
                                           load_untagged_loc_from_db=True,
                                           restrict_to_tagged_coords=False,
                                           restrict_to_tagged_locs=False,
                                           evt_types_to_load=EVENT_TYPES,
                                           evt_types_to_cmp=EVENT_TYPES,
                                           saving_path=experiment_default_path,
                                           plot_rocs=False):
    print('SOZ predictor in Whole Brain')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        str(load_untagged_coords_from_db)))
    print(
        'load_untagged_loc_from_db: {0}'.format(str(load_untagged_loc_from_db)))
    print(
        'restrict_to_tagged_coords: {0}'.format(str(restrict_to_tagged_coords)))
    print('restrict_to_tagged_locs: {0}'.format(str(restrict_to_tagged_locs)))
    print('saving_path: {0}'.format(saving_path))


    def load_patients_input():
        return load_patients(elec_collection, evt_collection, intraop,
                             loc_granularity=0,
                             locations=['Whole Brain'],
                             event_type_names=evt_types_to_load,
                             models_to_run=models_to_run,
                             load_untagged_coords_from_db=load_untagged_coords_from_db,
                             load_untagged_loc_from_db=load_untagged_loc_from_db,
                             restrict_to_tagged_coords=restrict_to_tagged_coords,
                             restrict_to_tagged_locs=restrict_to_tagged_locs)

    patients_by_loc = time_counter(load_patients_input)

    loc_name = first_key(patients_by_loc)
    patients_dic = patients_by_loc[loc_name]
    event_type_data_by_loc = dict()
    event_type_data_by_loc[loc_name] = dict()

    # Create info header
    info_saving_path = saving_path + '_info.txt'
    # Create parents dirs if they dont exist
    Path(info_saving_path).parent.mkdir(0o777, parents=True, exist_ok=True)
    with open(info_saving_path, "w+") as file:
        print('Info data for Event types to compare...', file=file)
        for event_type_names in evt_types_to_cmp:
            type_group_name = '+'.join(event_type_names)
            type_group_name = 'HFOs' if type_group_name == '+'.join(
                HFO_TYPES) else type_group_name
            print('\nInfo from: {n}'.format(n=type_group_name), file=file)
            event_type_data_by_loc[loc_name][type_group_name] = region_info(
                patients_dic,
                event_type_names)
            print_info(event_type_data_by_loc[loc_name][type_group_name],
                       file=file)

    if plot_rocs:
        graphics.event_rate_by_loc(event_type_data_by_loc,
                                   metrics=['pse', 'pnee', 'auc'],
                                   roc_saving_path=saving_path)

    return event_type_data_by_loc, patients_dic


# 3 ii
# Localized solo hay un baseline que es el que tiene taggeado los
# x,y,z y el loc porque tienen que tener la loc definida
# Se hace un llamado a la db por localizacion porque no hace falta
# Doesnt care of angles null, in ml we will zoom that.
# Only for HFOs, not Spikes
def pse_hfo_rate_auc_relation(elec_collection, evt_collection):
    event_type_data_by_loc, data_by_loc = \
        evt_rate_soz_pred_baseline_localized(elec_collection,
                                             evt_collection,
                                             intraop=False,
                                             load_untagged_coords_from_db=True,
                                             load_untagged_loc_from_db=True,
                                             restrict_to_tagged_coords=True,
                                             restrict_to_tagged_locs=True,
                                             evt_types_to_load=HFO_TYPES,
                                             evt_types_to_cmp=[[t] for
                                                               t in
                                                               HFO_TYPES],
                                             locations={
                                                 g: all_loc_names(g)
                                                 for g
                                                 in [2,3,5]},
                                             saving_dir=
                                             exp_save_path[3]['ii'][
                                                 'dir'],
                                             plot_rocs=False)

    graphics.plot_pse_hfo_rate_auc_table(data_by_loc, str(Path(exp_save_path[3][
                                                                   'ii']['dir'],
                                                               'table')))

    graphics.plot_co_pse_auc(data_by_loc, str(Path(exp_save_path[3][
                                                       'ii']['dir'],
                                                   'scatter')))


# TODO ver la cantidad de electrodos empty para hacer combinacion de tipos
# FIXME hacer adaptativos los cuadros de las figuras para q se guarden bien
# 3.iii
# saving_path aca es un directorio
def evt_rate_soz_pred_baseline_localized(elec_collection,
                                         evt_collection,
                                         intraop=False,
                                         load_untagged_coords_from_db=True,
                                         load_untagged_loc_from_db=True,
                                         restrict_to_tagged_coords=True,
                                         restrict_to_tagged_locs=True,
                                         evt_types_to_load=HFO_TYPES + [
                                             'Spikes'],
                                         evt_types_to_cmp=[[t] for
                                                           t in
                                                           HFO_TYPES + [
                                                               'Spikes']],
                                         locations={g: preference_locs(g) for g
                                                    in [2, 3, 5]},
                                         saving_dir=
                                         exp_save_path[3]['iii']['dir'],
                                         models_to_run=models_to_run,
                                         return_pat_dic_by_loc=False,
                                         plot_rocs=False):
    print('SOZ predictor localized Analysis')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(load_untagged_coords_from_db))
    print('load_untagged_loc_from_db: {0}'.format(load_untagged_loc_from_db))
    print('restrict_to_tagged_coords: {0}'.format(restrict_to_tagged_coords))
    print('restrict_to_tagged_locs: {0}'.format(restrict_to_tagged_locs))
    print('evt_types_to_load : {0}'.format(HFO_TYPES + [
        'Spikes']))
    print('evt_to_cmp: {0}'.format([[t] for t in HFO_TYPES + ['Spikes']]))
    print('locations: {0}'.format(locations))
    print('saving_dir: {0}'.format(saving_dir))
    print('models_to_run: {0}'.format(models_to_run))


    #patients_by_loc = None  # necessary? i think its not
    local_filter = True
    if local_filter:
        patients_by_loc = load_patients(elec_collection, evt_collection,
                                        intraop,
                                        loc_granularity=0,
                                        locations=['Whole Brain'],
                                        event_type_names=evt_types_to_load,
                                        models_to_run=models_to_run,
                                        load_untagged_coords_from_db=load_untagged_coords_from_db,
                                        load_untagged_loc_from_db=load_untagged_loc_from_db,
                                        restrict_to_tagged_coords=restrict_to_tagged_coords,
                                        restrict_to_tagged_locs=restrict_to_tagged_locs)

    # Saves pse and AUC ROC of each HFO type of baseline rate
    data_by_loc = dict()
    for granularity, locs in locations.items():
        if not local_filter:
            patients_by_loc = load_patients(elec_collection, evt_collection,
                                            intraop,
                                            loc_granularity=granularity,
                                            locations=locs,
                                            event_type_names=evt_types_to_load,
                                            models_to_run=models_to_run,
                                            load_untagged_coords_from_db=load_untagged_coords_from_db,
                                            load_untagged_loc_from_db=load_untagged_loc_from_db,
                                            restrict_to_tagged_coords=restrict_to_tagged_coords,
                                            restrict_to_tagged_locs=restrict_to_tagged_locs)

        event_type_data_by_loc = dict()
        for loc_name in locs:  # old cond: patients_dic in
            # patients_by_loc.items() works with location = None
            event_type_data_by_loc[loc_name] = dict()
            if local_filter:
                whole_brain_name = first_key(patients_by_loc)
                patients_dic = patients_by_loc[whole_brain_name]  # whole_brain_name

            else:
                patients_dic = patients_by_loc[loc_name]

            #for artifact_type in ['RonO', 'FronO']:
            #   patients_dic = kmean(artifact_type, patients_dic)
            # Create info header
            file_saving_path = str(Path(saving_dir,
                                        'loc_{g}'.format(g=granularity),
                                        loc_name.replace(' ', '_'),
                                        loc_name.replace(' ', '_') +
                                        '_sleep'))
            # Create parents dirs if they dont exist
            info_saving_path = file_saving_path + '_info.txt'
            Path(info_saving_path).parent.mkdir(0o777, parents=True,
                                                exist_ok=True)

            with open(info_saving_path, "w+") as file:
                print(
                    'Info data in ' + loc_name + ' for Event types to compare... ',
                    file=file)
                for event_type_names in evt_types_to_cmp:
                    type_group_name = '+'.join(event_type_names)
                    print('\nInfo from: {n}'.format(n=type_group_name),
                          file=file)
                    loc_info = region_info(patients_dic, event_type_names,
                                           location=loc_name if loc_name !=
                                                                'Whole Brain'
                                           else None)

                    min_pat_count_in_location = 12
                    min_pat_with_epilepsy_in_location = 3
                    if loc_info['patient_count'] >= min_pat_count_in_location \
                            and loc_info[
                        'patients_with_epilepsy'] >= min_pat_with_epilepsy_in_location:
                        #print('Files_saving_path {0}'.format(file_saving_path))

                        if loc_name not in data_by_loc.keys():
                            data_by_loc[loc_name] = dict()

                        # For ROCs plot
                        event_type_data_by_loc[loc_name][type_group_name] = \
                            loc_info

                        # Generate de txt file with useful data
                        print_info(loc_info, file=file)

                        # For exp 2
                        data_by_loc[loc_name][type_group_name + '_rates'] = \
                            dict(soz=loc_info['soz_rates'], nsoz=loc_info[
                                'nsoz_rates'])

                        # For 3.ii table
                        data_by_loc[loc_name]['PSE'] = loc_info['pse']
                        data_by_loc[loc_name][type_group_name + '_AUC'] = \
                            loc_info['AUC_ROC']

                        # For saving a location dictionary
                        # Same for each hfo type, whole dic
                        if return_pat_dic_by_loc:
                            data_by_loc[loc_name]['patients_dic'] = \
                                loc_info['patients_dic_in_loc']

                    else:
                        print(
                            'Region and type excluded because lack of data --> {0} {1}'.format(
                                loc_name, type_group_name))
        if plot_rocs:
            graphics.event_rate_by_loc(event_type_data_by_loc,
                                       metrics=['pse', 'pnee', 'auc'],
                                       roc_saving_path=str(Path(saving_dir,
                                                                'loc_{g}'.format(
                                                                    g=granularity),
                                                                'rate_baseline')),
                                       change_tab_path=True)
    return event_type_data_by_loc, data_by_loc


###################             Auxiliary functions             #########################################

# Gathers info about patients rate data for the types included in the list
# If loc is None all the dic is considered, otherwise only the location asked
def region_info(patients_dic, event_types=EVENT_TYPES, flush=False,
                conf=None, location=None):
    print('Region info location {0}, types {1}.'.format(location, event_types))
    patients_with_epilepsy = set()
    elec_count_per_patient = []
    elec_x_null, elec_y_null, elec_z_null = 0, 0, 0  # todo create dic
    elec_cnt_loc2_empty, elec_cnt_loc3_empty, elec_cnt_loc5_empty = 0, 0, 0  # todo create dic
    pat_with_x_null, pat_with_y_null, pat_with_z_null = set(), set(), set()  # todo create dic
    pat_with_loc2_empty, pat_with_loc3_empty, pat_with_loc5_empty = set(), set(), set()  # todo create dic
    soz_elec_count, elec_with_evt_count, event_count = 0, 0, 0
    counts = {type: 0 for type in event_types}
    event_rates, soz_labels, soz_rates, nsoz_rates = [], [], [], []

    if location is not None:
        patients_dic = {p_name: p for p_name, p in patients_dic.items() if \
                        p.has_elec_in(loc=location)}
    pat_in_loc = dict()

    for p_name, p in patients_dic.items():
        if location is None:
            electrodes = p.electrodes
            pat_in_loc[p_name] = p #was commented
        else:
            electrodes = [e for e in p.electrodes if getattr(e,
                                                             'loc{i}'.format(i=
                                                             get_granularity(
                                                                 location)))
                          == location]
            pat_in_loc[p_name] = Patient(p_name, p.age, electrodes)

        elec_count_per_patient.append(len(electrodes))
        assert (len(electrodes) > 0)
        for e in electrodes:
            if flush:
                e.flush_cache(event_types)
            if e.soz:
                patients_with_epilepsy.add(p_name)
                soz_elec_count = soz_elec_count + 1
            if e.x is None:
                elec_x_null += 1
                pat_with_x_null.add(p_name)
            if e.y is None:
                elec_y_null += 1
                pat_with_y_null.add(p_name)
            if e.z is None:
                elec_z_null += 1
                pat_with_z_null.add(p_name)
            if e.loc2 == 'empty':
                elec_cnt_loc2_empty += 1
                pat_with_loc2_empty.add(p_name)
            if e.loc3 == 'empty':
                elec_cnt_loc3_empty += 1
                pat_with_loc3_empty.add(p_name)
            if e.loc5 == 'empty':
                elec_cnt_loc5_empty += 1
                pat_with_loc5_empty.add(p_name)

            elec_evt_count = e.get_events_count(event_types)
            elec_with_evt_count = elec_with_evt_count + 1 if elec_evt_count > 0 else elec_with_evt_count
            event_count += elec_evt_count

            for type in event_types:
                counts[type] += e.get_events_count([type])

            evt_rate = e.get_events_rate(event_types)  # Measured in events/min
            event_rates.append(evt_rate)
            soz_labels.append(e.soz)

            if e.soz:
                soz_rates.append(evt_rate)
            else:
                nsoz_rates.append(evt_rate)

    elec_count = sum(elec_count_per_patient)
    pse = soz_elec_count / elec_count  # proportion of soz electrodes
    non_empty_elec_prop = elec_with_evt_count / elec_count
    try:
        auc_roc = roc_auc_score(soz_labels, event_rates)
    except ValueError:
        auc_roc = None
    info = {
        'patients_dic_in_loc': pat_in_loc,
        'patient_count': len(list(patients_dic.keys())),
        'patients_with_epilepsy': len(patients_with_epilepsy),
        'elec_count': elec_count,
        'mean_elec_per_pat': np.mean(elec_count_per_patient),
        'soz_elec_count': soz_elec_count,
        'pse': round(100 * pse, 2),  # percentage of SOZ electrodes
        'pnee': round(100 * non_empty_elec_prop, 2),
        # percentage of non empty elec
        'evt_count': event_count,  # Total count of all types
        'evt_count_per_type': counts,
        'AUC_ROC': auc_roc,
        # Baseline performance in region for these types collapsed
        'conf': conf,  # Sensibility and specificity for the simulator
        'pat_with_x_null': pat_with_x_null,
        'pat_with_y_null': pat_with_y_null,
        'pat_with_z_null': pat_with_z_null,
        'pat_with_loc2_empty': pat_with_loc2_empty,
        'pat_with_loc3_empty': pat_with_loc3_empty,
        'pat_with_loc5_empty': pat_with_loc5_empty,
        'elec_x_null': elec_x_null,
        'elec_y_null': elec_y_null,
        'elec_z_null': elec_z_null,
        'elec_cnt_loc2_empty': elec_cnt_loc2_empty,
        'elec_cnt_loc3_empty': elec_cnt_loc3_empty,
        'elec_cnt_loc5_empty': elec_cnt_loc5_empty,
        'evt_rates': event_rates,  # events per minute for each electrode
        'soz_labels': soz_labels,  # SOZ label of each electrode
        'soz_rates': soz_rates,
        'nsoz_rates': nsoz_rates,
    }
    return info


def first_key(dic):
    return [k for k in dic.keys()][0]

def print_roc_auc(labels, preds):
    fpr, tpr, threshold = roc_curve(labels, preds)
    assert(auc(fpr, tpr) == roc_auc_score(labels, preds))
    print('AUC --> {0}'.format(auc(fpr, tpr)))

'''
# VER SI VA EN ML_HFO_CLASSIFIER
# 6) pHFOs rate VS HFO rate baseline
# Tambien probe en vez de usar la prop de phfos > thresh en vez de hfo rate
# only added to test a classifier calc+ug baseline rate as feature

def Hippocampal_RonS_gradual_filters(elec_collection, evt_collection):
    model_name = 'XGBoost'
    models_to_run = [model_name]
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=tol_fprs, models_to_run=models_to_run,
                   comp_with='{0} '.format(model_name))


def ml_with_rate(elec_collection, evt_collection, loc_name, hfo_type_name,
                 tol_fprs):
    intraop = False
    model_name = 'XGBoost'
    models_to_run = [model_name]
    event_type_data_by_loc = {loc_name: {}}
    loc, locations = get_locations(5, [loc_name])
    target = ['model_pat']
    elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc,
                                            loc_name)
    elec_cursor = elec_collection.find(elec_filter,
                                       projection=electrodes_query_fields)
    hfo_cursor = evt_collection.find(evt_filter,
                                     projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor, models_to_run)
    print('Total patients {0}'.format(len(patients_dic)))

    for p in patients_dic.values():
        for e in p.electrodes:
            rate = e.get_events_rate([hfo_type_name])
            for h in e.events[hfo_type_name]:
                h.info['rate'] = rate

    target_patients = phfo_predictor(loc_name, hfo_type_name, patients_dic,
                                     target=target, models=models_to_run)

    all_target = 'model_pat' in target and 'validation_pat' in target
    if all_target:
        name = ' baseline all'
    if not all_target:
        name = ' baseline {0}'.format(target[0])
    event_type_data_by_loc[loc_name][hfo_type_name + name] = region_info(
        {p.id: p for p in target_patients}, [hfo_type_name])

    print('Running model {0}'.format(model_name))
    labels, preds, probs = gather_folds(model_name, hfo_type_name,
                                        target_patients=target_patients)
    print('Displaying metrics for phfo classifier')
    print_metrics(model_name, hfo_type_name, labels, preds, probs)

    # SOZ HFO RATE MODEL
    fpr, tpr, thresholds = roc_curve(labels, probs)
    for tol_fpr in tol_fprs:
        thresh = get_soz_confidence_thresh(fpr, thresholds,
                                           tolerated_fpr=tol_fpr)  # if the prob is more than this thresh I will consider it for the mean
        filtered_pat_dic = phfo_thresh_filter(target_patients,
                                              hfo_type_name,
                                              thresh=thresh, perfect=False,
                                              model_name=model_name)

        rated_data = {
            'evt_rates': [],  # mean probs
            'soz_labels': []
        }
        elec_count = 0
        for patient in filtered_pat_dic.values():
            elec_count += len(patient.electrodes)
            for e in patient.electrodes:
                filtered_probs = [h.info['proba'][model_name] for h in
                                  e.events[hfo_type_name]]
                mean_prob = np.mean(filtered_probs) if len(
                    filtered_probs) > 0 else 0
                rated_data['evt_rates'].append(mean_prob)
                rated_data['soz_labels'].append(e.soz)

        rated_data['elec_count'] = elec_count
        rated_data['AUC_ROC'] = roc_auc_score(rated_data['soz_labels'],
                                              rated_data['evt_rates'])
        event_type_data_by_loc[loc_name][
            hfo_type_name + model_name + ' mean prob ' + ' FPR {0}'.format(
                tol_fpr)] = rated_data

    graphics.event_rate_by_loc(event_type_data_by_loc, metrics=['auc'],
                               title='Hippocampal RonS HFO rate (events per minute) baseline \nVS ML rate, x, y, z, properties.',
                               colors=None)

def Hippocampal_RonS_ml_with_rate(elec_collection, evt_collection, ):
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_with_rate(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                 tol_fprs=tol_fprs)



# The two main functions that compare baseline with ml
def compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    intraop=False, loc_granularity=0,
                                    locations='all',
                                    event_type_names=EVENT_TYPES,
                                    filter_phfos=False, filter_info=None,
                                    bs_info_by_loc=None,
                                    saving_path=EXPERIMENTS_FOLDER):
    print('Comparing event type rates by location...')
    if filter_phfos:
        assert (all(
            [isinstance(filter_info, dict), 'target' in filter_info.keys(),
             'tol_fpr' in filter_info.keys()]))

    patients_by_loc = load_patients(elec_collection, evt_collection,
                                    intraop,
                                    loc_granularity, locations,
                                    event_type_names, models_to_run)

    event_type_data_by_loc = dict()
    print('Populating rate data by location and type...')
    for loc_name, patients_dic in patients_by_loc.items():
        event_type_data_by_loc[loc_name] = dict()
        for evt_type_name in event_type_names:
            event_type_data_by_loc[loc_name][evt_type_name] = region_info(
                patients_dic, [evt_type_name])
            if bs_info_by_loc is not None:
                bs_info_by_loc[loc_name][evt_type_name] = \
                    event_type_data_by_loc[loc_name][evt_type_name][
                        'AUC_ROC']
                bs_info_by_loc[loc_name]['PSE'] = \
                    event_type_data_by_loc[loc_name][evt_type_name][
                        'pse']  # Should agree among types, checked in db_parsing

            if filter_phfos and evt_type_name not in ['Spikes',
                                                      'Sharp Spikes']:
                patients_dic = phfo_filter(evt_type_name, patients_dic,
                                           target=filter_info['target'],
                                           tolerated_fpr=filter_info[
                                               'tol_fpr'],
                                           perfect=filter_info['perfect'])
                event_type_data_by_loc[loc_name][
                    'Filtered ' + evt_type_name] = region_info(patients_dic,
                                                               [
                                                                   evt_type_name],
                                                               flush=True)

    graphics.event_rate_by_loc(event_type_data_by_loc, saving_path)


def compare_subtypes_rate_by_loc(elec_collection, evt_collection,
                                 hfo_type_name,
                                 subtypes='all', loc_granularity=0,
                                 locations='all',
                                 intraop=False, filter_phfos=False,
                                 filter_info=None,
                                 saving_path=EXPERIMENTS_FOLDER):
    # In this case we need to load subtype and calculate the rate to avoid mixing the subtypes in the rate.
    # The structure doesn't differ among subtypes
    subtypes = all_subtype_names(
        hfo_type_name) if subtypes == 'all' else subtypes
    loc, locations = get_locations(loc_granularity, locations)
    subtype_data_by_loc = dict()
    for loc_name in locations:
        subtype_data_by_loc[loc_name] = dict()
        for subtype_name in subtypes:
            elec_filter, hfo_filter = query_filters(intraop,
                                                    [hfo_type_name],
                                                    loc, loc_name,
                                                    [subtype_name])
            elec_cursor = elec_collection.find(elec_filter,
                                               projection=electrodes_query_fields)
            hfo_cursor = evt_collection.find(hfo_filter,
                                             projection=hfo_query_fields)
            patients_dic = parse_patients(elec_cursor, hfo_cursor,
                                          models_to_run)
            subtype_data_by_loc[loc_name][subtype_name] = region_info(
                patients_dic, [hfo_type_name])

            if filter_phfos:
                patients_dic = phfo_filter(hfo_type_name, patients_dic,
                                           target=filter_info['target'],
                                           tolerated_fpr=filter_info[
                                               'tol_fpr'],
                                           perfect=filter_info['perfect'])
                subtype_data_by_loc[loc_name][
                    'Filtered_' + hfo_type_name] = region_info(patients_dic,
                                                               [
                                                                   hfo_type_name])

    graphics.event_rate_by_loc(subtype_data_by_loc,
                               zoomed_type=hfo_type_name,
                               roc_saving_path=saving_path)
    plt.show()
'''
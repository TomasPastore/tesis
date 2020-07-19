from sys import version as py_version
import warnings
from pathlib import Path

from patient import Patient

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from sklearn.metrics import roc_auc_score
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
                                           saving_path=experiment_default_path):
    print('SOZ predictor in Whole Brain')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        str(load_untagged_coords_from_db)))
    print(
        'load_untagged_loc_from_db: {0}'.format(str(load_untagged_loc_from_db)))
    print(
        'restrict_to_tagged_coords: {0}'.format(str(restrict_to_tagged_coords)))
    print('restrict_to_tagged_locs: {0}'.format(str(restrict_to_tagged_locs)))

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
                                                 in [2]},
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
                                         loc_pat_dic=None,
                                         plot_rocs=True):
    print('SOZ predictor localized')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        str(load_untagged_coords_from_db)))
    print(
        'load_untagged_loc_from_db: {0}'.format(str(load_untagged_loc_from_db)))
    print(
        'restrict_to_tagged_coords: {0}'.format(str(restrict_to_tagged_coords)))
    print('restrict_to_tagged_locs: {0}'.format(str(restrict_to_tagged_locs)))
    '''
    #Prints all locations
    print(elec_collection.distinct('loc2'))
    print(elec_collection.distinct('loc3'))
    print(elec_collection.distinct('loc5'))
    print(evt_collection.distinct('loc2'))
    print(evt_collection.distinct('loc3'))
    print(evt_collection.distinct('loc5'))
    '''
    patients_by_loc = None  # necessary? i think its not
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
                patients_dic = patients_by_loc[
                    whole_brain_name]  # whole_brain_name
            else:
                patients_dic = patients_by_loc[loc_name]
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
                        print('Files_saving_path {0}'.format(file_saving_path))

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
                        if loc_name == loc_pat_dic:
                            data_by_loc[loc_pat_dic]['patients_dic'] = \
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
            # pat_in_loc[p_name] = p
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

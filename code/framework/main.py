import copy
import random
import sys
from sys import version as py_version
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import \
    pyplot as plt  # todo ver donde se usa, deberia estar solo en graphics
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon, ranksums, kstest, ks_2samp
import math as mt
from datetime import timedelta
from config import (EVENT_TYPES, HFO_TYPES, exp_save_path,
                    EXPERIMENTS_FOLDER, experiment_default_path,
                    intraop_patients, non_intraop_patients,
                    electrodes_query_fields, hfo_query_fields, models_to_run)
from db_parsing import Database, parse_patients, get_locations, \
    encode_type_name, all_loc_names, load_patients, query_filters, \
    all_subtype_names
from metrics import print_metrics
from ml_hfo_classifier import phfo_filter, phfo_predictor, gather_folds, \
    get_soz_confidence_thresh, phfo_thresh_filter

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    from utils import histograms, phase_coupling_paper_polar
from utils import all_subsets, LOG, print_info
import tests
import graphics


# TODO move to main and pass collections by parameters

def main():
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    #Thesis experiments
    run_experiment(elec_collection, evt_collection, number=3, roman_num='i',
                   letter='a')

    # TODO week 26/7 dump to overleaf reeplazing commented structure to code structure with if
    # Trabajo futuro combinacion de subtipos, mas datos,

def run_experiment(elec_collection, evt_collection, number, roman_num,
                   letter):
    NOT_IMPLEMENTED_EXP = NotImplementedError('Unknown experiment class')
    if number == 0:
        print('Runing data loading and tests')
        # Set exit parameter to True if you just want the tests and comment
        # if you don't want to test before computing
        # TODO Agregar a informe, tests
        tests.unittest.main(exit=False)
    elif number == 1:
        print('Running exp 1) Data Global analysis')
        # Total patient count, intraop vs Non intraop, other info
        # IO have artifacts, discarded ? TODO ask shennan
        # TODO correr con el nuevo info y agregar a informe
        # global_info_in_loc(elec_collection, evt_collection,
        #                     intraop=False, loc_granularity = 0,
        #                     locations = 'all', event_type_names = EVENT_TYPES)
        # show_patients_by_epilepsy_loc(soz_restricted = ['Temporal Lobe'],
        # soz_required=['Temporal Lobe'])
        raise NOT_IMPLEMENTED_EXP
    elif number == 2:
        if roman_num == 'i':
            print('Running exp 2.i) HFO rate in SOZ vs NSOZ in Whole brain')
            raise NOT_IMPLEMENTED_EXP
        elif roman_num == 'ii':
            print('Running exp 2.ii) HFO rate in SOZ vs NSOZ localized')
            raise NOT_IMPLEMENTED_EXP
        else: # roman_num
            raise NOT_IMPLEMENTED_EXP
    elif number == 3:
        # Se quiere mejorar el rendimiento del rate de los distintos tipos
        # de HFO, para eso veamos los baselines
        # TODO move to soz_predictor module
        if roman_num == '0':
            print('Running exp 3.0) Predicting SOZ with rates: Baselines '
                  '(first steps)')
            # TODO add to overleaf
            evt_rate_soz_pred_baseline_whole_brain(elec_collection,
                                                   evt_collection,
                                                   intraop=False,
                                                   load_untagged_coords_from_db=True,
                                                   load_untagged_loc_from_db=True,
                                                   restrict_to_tagged_coords=False,
                                                   restrict_to_tagged_locs=False,
                                                   evt_types_to_load=EVENT_TYPES,
                                                   evt_types_to_cmp=[
                                                       HFO_TYPES, ['Spikes']],
                                                   saving_path=exp_save_path[3][
                                                       '0'])
        elif roman_num == 'i':
            # TODO add to overleaf
            if letter == 'a':
                print('Running exp 3.i.a) Predicting SOZ with rates: '
                    'Baselines (Whole brain coords untagged)')
                # Ma)ML with 91 patients without using coords (untagged)
                # Whole brain rates for independent event types:
                evt_rate_soz_pred_baseline_whole_brain(elec_collection,
                                                   evt_collection,
                                                   intraop=False,
                                                   load_untagged_coords_from_db=True,
                                                   load_untagged_loc_from_db=True,
                                                   restrict_to_tagged_coords=False,
                                                   restrict_to_tagged_locs=False,
                                                   evt_types_to_load=HFO_TYPES+['Spikes'],
                                                   evt_types_to_cmp=[[t] for
                                                                     t in HFO_TYPES+['Spikes']],
                                                   saving_path=exp_save_path[
                                                       3]['i']['a'])
            elif letter == 'b':
                print('Running exp 3.i.B) Predicting SOZ with rates: '
                      'Baselines (Whole brain coords tagged)')
                # Mb) 57 patients with tagged coords.
                # NOTE: V0_ if you just dont load untagged coords from db,
                # it improves AUC 2%, we loose 34 patients, but there are
                # still 2 electrodes in None because of bad format of
                # the field (empty lists map to None) from db
                evt_rate_soz_pred_baseline_whole_brain(elec_collection,
                                                       evt_collection,
                                                       intraop=False,
                                                       load_untagged_coords_from_db=True,
                                                       load_untagged_loc_from_db=True,
                                                       restrict_to_tagged_coords=True,
                                                       restrict_to_tagged_locs=False,
                                                       evt_types_to_load=HFO_TYPES + [
                                                           'Spikes'],
                                                       evt_types_to_cmp=[[t] for
                                                                     t in HFO_TYPES+['Spikes']],
                                                       saving_path=
                                                       exp_save_path[
                                                           3]['i']['b'])
            else:  # letter
                raise NOT_IMPLEMENTED_EXP
        elif roman_num == 'ii':
            print('Running exp 3.ii.a) Predicting SOZ with rates: '
                  'Baselines (Localized x,y,z and loc tagged)')
            # TODO agregar spikes y revisar all zones
            # Localized solo hay un baseline que es el que tiene taggeado los
            # x,y,z y el loc porque tienen que tener la loc definida
            # Se carga toda la base, se resuelve inconsitencias y
            # luego se clasifica por loc del electrodo.
            # TODO now
            raise NOT_IMPLEMENTED_EXP
            hfo_types_in_locations(elec_collection, evt_collection,
                                   allow_null_coords_db=True,
                                   allow_empty_loc_db=True, rm_xyz_null=True,
                                   rm_loc_empty=True)
        elif roman_num == 'iii':
            # TODO 4/7
            print('Running exp 3.iii: Proportion of soz electrodes AUC '
                  'relation')
            raise NOT_IMPLEMENTED_EXP
            # pse_hfo_rate_auc_relation(elec_collection, evt_collection)
        else:  # roman_num
            raise NOT_IMPLEMENTED_EXP
    elif number == 4:
        # TODO week 6/7
        raise NOT_IMPLEMENTED_EXP
        # 4) ML HFO classifiers para extremos de 3 iii
        # Usar sklearn pipeline
        # Resultados: xgboost, model_patients (%75), random partition, robust scaler, balanced, filter 0.7 da 0.8 de AP
        # Statistical_tests(elec_collection, evt_collection) # TODO sacar de aca
        # whole_brain_hfo_classifier(elec_collection, evt_collection) # TODO with allowed and not allowed coords
        # hippocampus_hfo_classifier(elec_collection, evt_collection) # TODO with allowed or not allowed coords if loc makes this possible
        # phfo_analysis_zoom(elec_collection, evt_collection,'Hippocampus', 'RonS') #TODO meter adentro del analisis del hipocampo para el ml
        # Frons in hippocampuss
    elif number == 5:
        # TODO week 12/7
        raise NOT_IMPLEMENTED_EXP
        # 5) pHFOs rate VS HFO rate baseline
        # phfo_rate_vs_baseline_whole_brain(elec_collection, evt_collection, allow_null_coords=True, event_type_names) #TODO
        # phfo_rate_vs_baseline_whole_brain(elec_collection, evt_collection, allow_null_coords=True) #TODO

        # compare_Hippocampal_RonS_ml_models(elec_collection, evt_collection)
        # Hippocampal_RonS_gradual_filters(elec_collection, evt_collection)
        # Hippocampal_RonS_ml_with_rate(elec_collection, evt_collection) # TODO mencionar en discusion de resultados de la comparacion de baseline vs ml filters
    elif number == 6:
        # TODO week 19/7
        raise NOT_IMPLEMENTED_EXP
        # 6) Simulation of the ml predictor to understand needed performance to
        # improve HFO rate baseline
        # simulator(elec_collection, evt_collection) TODO mencionar en discusion de resultados de la comparacion de baseline vs ml filters
    else: # number
        raise NOT_IMPLEMENTED_EXP

############################       Development Steps     #######################################

# 1) Data Global analysis  #####################################################################
# Electrodes collection didn't have intraop field, we got all the patients from all the events
def print_non_intraop_patients():
    print(intraop_patients)
    print('Count: {0}'.format(len(intraop_patients)))


# We developed this function that gathers global info about the region, it will be used for localized analysis later
def global_info_in_locations(elec_collection, evt_collection, intraop=False,
                             loc_granularity=0, locations='all',
                             event_type_names=EVENT_TYPES):
    print('Gathering global info...')
    patients_by_loc = load_patients(elec_collection, evt_collection, intraop,
                                    loc_granularity, locations,
                                    event_type_names, models_to_run,
                                    allow_null_coords=True,
                                    allow_empty_loc=True)
    for loc_name in patients_by_loc.keys():
        patients_dic = patients_by_loc[loc_name]
        info = region_info(patients_dic, event_types=event_type_names,
                           flush=False, conf=None)
        print('Global info in location: {0} \n {1}'.format(loc_name, info))


# TODO view how many patients with focal epilepsy per locations do we have and evaluate null elements
def show_patients_by_epilepsy_loc(elec_collection, evt_collection,
                                  intraop=False, loc_granularity=0,
                                  locations='all',
                                  event_type_names=EVENT_TYPES,
                                  soz_restricted=[''], soz_required=None):
    patients_by_loc = load_patients(elec_collection, evt_collection, intraop,
                                    loc_granularity, locations,
                                    event_type_names,
                                    models_to_run, allow_null_coords=True,
                                    allow_empty_loc=True)
    loc_name = [loc for loc in patients_by_loc.keys()][0]
    patients_dic = patients_by_loc[loc_name]

    if soz_required is None:
        soz_required = []

    comb_by_granularity = {0: ['All brain']}
    for granularity in range(1, 6):
        comb_by_granularity[granularity] = []

        loc_field, all_locations = get_locations(granularity, 'all')
        all_locations.append(
            'empty')  # empty represents patients who have empty loc in one electrode of this granularity
        for comb in all_subsets(
                all_locations):  # returns all subsets except from empty set
            comb_by_granularity[granularity].append(comb)

    print(
        'Count of patients with epilepsy restricted to regions by granularity')
    for granularity, combs in comb_by_granularity.items():
        for comb in combs:
            comb_count = {comb: 0}
            comb_patients = {comb: []}
            for pat in patients_dic.values():
                if pat.has_epilepsy_in_all_locs(granularity,
                                                soz_required):  # pat.has_epilepsy_restricted_to(granularity, comb) and
                    comb_count[comb] += 1
                    comb_patients[comb].append(pat.id)
            if comb_count[comb] > 0:
                if 'Temporal Lobe' in comb:
                    print(
                        '(Granularity: {0}, Epilepsy_spots_restricted_to: {1}) --> #{2}. Patients: {3}'.format(
                            granularity, comb, comb_count[comb],
                            comb_patients[comb]))


# 2) HFO rate in SOZ vs NSOZ  ##################################################################
# Note: HFO rate is defined in classes.py module as a method for the electrode object
# TODO implementar
# Tengo Histograma de hfo rate de soz vs nsoz (green vs red),
# TODO review and remove, get histogram code for 3)
def phfo_analysis_zoom(electrodes_collection, hfo_collection, loc_name,
                       hfo_type_name):
    intraop = False
    loc, locations = get_locations(5, [loc_name])
    elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc,
                                            loc_name)
    elec_cursor = electrodes_collection.find(elec_filter,
                                             projection=electrodes_query_fields)
    hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor, models_to_run)
    red, green, orange, yellow, all_p = [], [], [], [], []
    sozs, p_proportions, pevt_counts = [], [], []
    empties, with_hfos, all_empties = [], [], []
    red_rates, green_rates = [], []
    for p in patients_dic.values():
        for e in p.electrodes:
            soz = e.soz
            sozs.append(soz)
            phfo = e.has_pevent([hfo_type_name])
            if phfo:
                pevt_counts.append(e.pevt_count[hfo_type_name])

            prop_score, empty = e.pevent_proportion_score([hfo_type_name])
            all_empties.append(empty)
            if empty:
                empties.append(prop_score)
            else:
                with_hfos.append(prop_score)
            p_proportions.append(prop_score)
            if soz and phfo:  # canal soz con al menos un phfo
                red_rates.append(e.get_events_rate([hfo_type_name]))
                red.append(prop_score)
                all_p.append(prop_score)
            elif not soz and not phfo:
                rate = e.get_events_rate([hfo_type_name])
                green_rates.append(rate)
                green.append(prop_score)
                if rate > 0:
                    all_p.append(prop_score)

            elif soz and not phfo:
                orange.append(prop_score)
                if len(e.events[hfo_type_name]) > 0:
                    all_p.append(prop_score)
            elif not soz and phfo:  # Que no haya amarillos implica que phfo implica soz
                yellow.append(prop_score)

    print(
        'Orange (soz and not phfo) proportions (if 1 means that they dont have any hfo, if 0 means that it has fhfos): {0}'.format(
            orange))

    # TODOS LOS ELECTRODOS QUE TIENEN UN PHFO SON SOZ...  en hippocampo
    print('Proportion by label...')
    for i in range(len(sozs)):
        print('SOZ: {0}'.format(sozs[i]))
        print('Empty: {0}'.format(all_empties[i]))
        print('Phfo proportion: {0}'.format(p_proportions[i]))

    print('With hfos props')
    print(with_hfos)
    print('Empty electrodes props')
    print(empties)
    elec_count = len(with_hfos) + len(empties)
    print('Total elec count {0}'.format(elec_count))
    print('Empty proportion {0}'.format(len(empties) / elec_count))

    graphics.barchart(len(red), len(green), len(yellow), len(orange))
    graphics.histogram(all_p,
                       title='{0} in {1} pathologic proportion per electrode'.format(
                           hfo_type_name, loc_name),
                       x_label='Pathologic proportion')  # empty electrodes were excluded

    graphics.histogram(red,
                       title='{0} in {1} pathologic proportion per electrode (RED elec)'.format(
                           hfo_type_name, loc_name),
                       x_label='Pathologic proportion')

    graphics.hfo_rate_histogram_red_green(red_rates, green_rates,
                                          title='HFO rate in SOZ vs NSOZ',
                                          bins=160)
    # graphics.histogram(pevt_counts, title='{0} in {1} pathologic event count per electrode'.format(hfo_type_name, loc_name),
    #                   x_label='Pathologic count', bins=np.arange(0,2800, 50))


# 3) Predicting SOZ with rates: Baselines  #####################################
def evt_rate_soz_pred_baseline_whole_brain(elec_collection, evt_collection,
                                           intraop=False,
                                           load_untagged_coords_from_db=True,
                                           load_untagged_loc_from_db=True,
                                           restrict_to_tagged_coords=False,
                                           restrict_to_tagged_locs=False,
                                           evt_types_to_load=EVENT_TYPES,
                                           evt_types_to_cmp=EVENT_TYPES,
                                           saving_path=experiment_default_path):
    print('SOZ predictor in whole brain')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        str(load_untagged_coords_from_db)))
    print(
        'load_untagged_loc_from_db: {0}'.format(str(load_untagged_loc_from_db)))
    print(
        'restrict_to_tagged_coords: {0}'.format(str(restrict_to_tagged_coords)))
    print('restrict_to_tagged_locs: {0}'.format(str(restrict_to_tagged_locs)))

    patients_by_loc = load_patients(elec_collection, evt_collection, intraop,
                                    loc_granularity=0,
                                    locations=['Whole Brain'],
                                    event_type_names=evt_types_to_load,
                                    models_to_run=models_to_run,
                                    allow_null_coords=load_untagged_coords_from_db,
                                    allow_empty_loc=load_untagged_loc_from_db,
                                    rm_xyz_null=restrict_to_tagged_coords,
                                    rm_loc_empty=restrict_to_tagged_locs)

    loc_name = first_key(patients_by_loc)
    patients_dic = patients_by_loc[loc_name]
    event_type_data_by_loc = dict()
    event_type_data_by_loc[loc_name] = dict()

    # Create info header
    info_saving_path = saving_path + '_info.txt'
    #Create parents dirs if they dont exist
    Path(info_saving_path).parent.mkdir(parents=True, exist_ok=True)
    with open(info_saving_path, "w+") as file:
        file.write('Info data for Event types to compare... \n')
        for event_type_names in evt_types_to_cmp:
            type_group_name = '+'.join(event_type_names)
            print('\nInfo from: {n}'.format(n=type_group_name), file=file)
            event_type_data_by_loc[loc_name][type_group_name] = region_info(
                                                                patients_dic,
                                                                event_type_names)
            print_info(event_type_data_by_loc[loc_name][type_group_name],
                        file=file)

    print('Plotting...')
    graphics.event_rate_by_loc(event_type_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               saving_path=saving_path)
    plt.show()


# TODO merge and add spikes
def hfo_types_in_locations(elec_collection, evt_collection,
                           allow_null_coords_db=True,
                           allow_empty_loc_db=True, rm_xyz_null=True,
                           rm_loc_empty=False):
    locations = ['Whole brain'] + all_loc_names(2) + all_loc_names(
        3) + all_loc_names(5)
    # Saves pse and AUC ROC of each HFO type of baseline rate
    columns = {c: 0 for c in ['PSE', HFO_TYPES]}
    baseline_info_by_loc = {loc: copy.deepcopy(columns) for loc in locations}
    compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    loc_granularity=0,
                                    event_type_names=HFO_TYPES,
                                    bs_info_by_loc=baseline_info_by_loc,
                                    saving_path=EXPERIMENTS_FOLDER)
    compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    loc_granularity=2,
                                    event_type_names=HFO_TYPES,
                                    bs_info_by_loc=baseline_info_by_loc,
                                    saving_path=EXPERIMENTS_FOLDER)
    compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    loc_granularity=3,
                                    event_type_names=HFO_TYPES,
                                    bs_info_by_loc=baseline_info_by_loc,
                                    saving_path=EXPERIMENTS_FOLDER)
    compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    loc_granularity=5,
                                    event_type_names=HFO_TYPES,
                                    bs_info_by_loc=baseline_info_by_loc,
                                    saving_path=EXPERIMENTS_FOLDER)

    # TODO merge
    # evt_rate_soz_predictor_hfo_types_vs_spikes_whole_brain(elec_collection, evt_collection, allow_null_coords=True,
    #                                                       allow_empty_loc=True, rm_xyz_null=True, rm_loc_empty=False)


def pse_hfo_rate_auc_relation(elec_collection, evt_collection):
    pass


# graphics.plot_score_table(bs_info_by_loc)
# graphics.plot_co_metrics_auc(tables['proportion'], tables['pse'], tables['AUC_ROC'])
# graphics.plot_co_metric_auc_0(tables['pscore'], tables['AUC_ROC'])


# 5) ML HFO classifiers
# Compare modelos con y sin balanceo, el scaler, la forma de hacer la particion de pacientes y param tuning
# Model patients da peor

def Statistical_tests(elec_collection, evt_collection):
    loc_name = 'Hippocampus'
    hfo_type_name = 'RonS'
    intraop = False
    model_name = 'XGBoost'
    models_to_run = [model_name]
    loc, locations = get_locations(5, [loc_name])
    elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc,
                                            loc_name)
    elec_cursor = elec_collection.find(elec_filter,
                                       projection=electrodes_query_fields)
    hfo_cursor = evt_collection.find(evt_filter, projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor, models_to_run)

    duration = {'soz': [], 'nsoz': []}
    freq_av = {'soz': [], 'nsoz': []}
    power_av = {'soz': [], 'nsoz': []}
    sin_spike_angle = {'soz': [], 'nsoz': []}
    cos_spike_angle = {'soz': [], 'nsoz': []}

    soz_count = 0
    nsoz_count = 0

    for p in patients_dic.values():
        for e in p.electrodes:
            for h in e.events[hfo_type_name]:
                if h.info['soz']:
                    soz_count += 1
                    duration['soz'].append(h.info['duration'])
                    freq_av['soz'].append(h.info['freq_av'])
                    power_av['soz'].append(h.info['power_av'])
                    sin_spike_angle['soz'].append(mt.sin(h.info['spike_angle']))
                    cos_spike_angle['soz'].append(mt.cos(h.info['spike_angle']))

                else:
                    nsoz_count += 1
                    duration['nsoz'].append(h.info['duration'])
                    freq_av['nsoz'].append(h.info['freq_av'])
                    power_av['nsoz'].append(h.info['power_av'])
                    sin_spike_angle['nsoz'].append(
                        mt.sin(h.info['spike_angle']))
                    cos_spike_angle['nsoz'].append(
                        mt.cos(h.info['spike_angle']))

    duration_statistic, duration_pval = ks_2samp(duration['soz'],
                                                 duration['nsoz'])
    freq_av_statistic, freq_av_pval = ks_2samp(freq_av['soz'], freq_av['nsoz'])
    power_av_statistic, power_av_pval = ks_2samp(power_av['soz'],
                                                 power_av['nsoz'])
    sin_spike_angle_statistic, sin_spike_angle_pval = ks_2samp(
        sin_spike_angle['soz'], sin_spike_angle['nsoz'])
    cos_spike_angle_statistic, cos_spike_angle_av_pval = ks_2samp(
        cos_spike_angle['soz'], cos_spike_angle['nsoz'])

    print('Duration Kolmogorov-Smirnov statistic, pvalue  : {0}, {1}'.format(
        duration_statistic, duration_pval))
    print('Frequency Kolmogorov-Smirnov statistic, pvalue  : {0}, {1}'.format(
        freq_av_statistic, freq_av_pval))
    print('Power Kolmogorov-Smirnov statistic, pvalue  : {0}, {1}'.format(
        power_av_statistic, power_av_pval))
    print(
        'Sin spike angle Kolmogorov-Smirnov statistic, pvalue  : {0}, {1}'.format(
            sin_spike_angle_statistic, sin_spike_angle_pval))
    print(
        'Cos spike angle Kolmogorov-Smirnov statistic, pvalue  : {0}, {1}'.format(
            cos_spike_angle_statistic, cos_spike_angle_av_pval))


def ml_phfo_models(elec_collection, evt_collection, loc_name, hfo_type_name,
                   tol_fprs, models_to_run=models_to_run, comp_with='',
                   conf=None):
    intraop = False
    event_type_data_by_loc = {loc_name: {}}
    loc, locations = get_locations(5, [loc_name])
    target = [
        'model_pat']  # this should be a parameter it can be ['model_pat'], ['validation_pat'],['model_pat', 'validation_pat']
    elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc,
                                            loc_name)
    elec_cursor = elec_collection.find(elec_filter,
                                       projection=electrodes_query_fields)
    hfo_cursor = evt_collection.find(evt_filter, projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor, models_to_run)
    print('Total patients {0}'.format(len(patients_dic)))

    target_patients = phfo_predictor(loc_name, hfo_type_name, patients_dic,
                                     target=target, models=models_to_run,
                                     conf=conf)

    all_target = 'model_pat' in target and 'validation_pat' in target
    if all_target:
        name = ' baseline all'
    if not all_target:
        name = ' baseline {0}'.format(target[0])
    event_type_data_by_loc[loc_name][hfo_type_name + name] = region_info(
        {p.id: p for p in target_patients}, [hfo_type_name])

    for model_name in models_to_run:
        simulating = conf is not None
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
                                                  hfo_type_name, thresh=thresh,
                                                  perfect=False,
                                                  model_name=model_name)
            confidence = None if model_name != 'Simulated' else conf
            print(
                'For model_name {0} conf is {1}'.format(model_name, confidence))
            rated_data = region_info(filtered_pat_dic, [hfo_type_name],
                                     flush=True,
                                     conf=confidence)  # calcula la info para la roc con la prob asociada al fpr tolerado
            event_type_data_by_loc[loc_name][
                hfo_type_name + ' hfo rate ' + model_name + ' FPR {0}'.format(
                    tol_fpr)] = rated_data

    graphics.event_rate_by_loc(event_type_data_by_loc, metrics=['ec', 'auc'],
                               title='Hippocampal RonS HFO rate (events per minute) baseline \nVS {0}filtered rate.'.format(
                                   comp_with),
                               colors='random' if comp_with == '' else None,
                               conf=conf)


# 6) pHFOs rate VS HFO rate baseline
# Tambien probe en vez de usar la prop de phfos > thresh en vez de hfo rate
# only added to test a classifier using baseline rate as feature
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
    hfo_cursor = evt_collection.find(evt_filter, projection=hfo_query_fields)
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
        filtered_pat_dic = phfo_thresh_filter(target_patients, hfo_type_name,
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


def compare_Hippocampal_RonS_ml_models(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Random Forest', 'Bayes']
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=[0.6], models_to_run=models_to_run)


def Hippocampal_RonS_gradual_filters(elec_collection, evt_collection, ):
    model_name = 'XGBoost'
    models_to_run = [model_name]
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=tol_fprs, models_to_run=models_to_run,
                   comp_with='{0} '.format(model_name))


def Hippocampal_RonS_ml_with_rate(elec_collection, evt_collection, ):
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_with_rate(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                 tol_fprs=tol_fprs)


# TODO move to simulator module
# 7) Simulation of the ml predictor to understand needed performance to improve HFO rate baseline
def simulator(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Simulated']
    for conf in [0.6, 0.7, 0.8,
                 0.9]:  # confianzas del simulador una antes y una despues de baseline
        comp_with = '{0} Simulator '.format(conf)
        print('Conf: {0}'.format(conf))
        tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                       tol_fprs=tol_fprs, models_to_run=models_to_run,
                       comp_with=comp_with, conf=conf)


###################             Auxiliary functions             #########################################

# Gathers info about patients rate data for the types included in the list
def region_info(patients_dic, event_types=EVENT_TYPES, flush=False, conf=None):
    print('Gathering info for types {0}.'.format(event_types))
    patients_with_epilepsy = set()
    elec_count_per_patient = []
    elec_x_null, elec_y_null, elec_z_null = 0, 0, 0  # todo create dic
    elec_cnt_loc2_empty, elec_cnt_loc3_empty, elec_cnt_loc5_empty = 0, 0, 0  # todo create dic
    pat_with_x_null, pat_with_y_null, pat_with_z_null = set(), set(), set()  # todo create dic
    pat_with_loc2_empty, pat_with_loc3_empty, pat_with_loc5_empty = set(), set(), set()  # todo create dic
    soz_elec_count, elec_with_evt_count, event_count = 0, 0, 0
    counts = {type: 0 for type in event_types}
    event_rates, soz_labels = [], []
    for p_name, p in patients_dic.items():
        elec_count_per_patient.append(len(p.electrodes))
        for e in p.electrodes:
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
                for fileblock, count in e.evt_count[type].items():
                    counts[type] += count

            event_rates.append(
                e.get_events_rate(event_types))  # Measured in events/min
            soz_labels.append(e.soz)

    elec_count = sum(elec_count_per_patient)
    pse = soz_elec_count / elec_count  # proportion of soz electrodes
    non_empty_elec_prop = elec_with_evt_count / elec_count
    info = {
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
        'AUC_ROC': roc_auc_score(soz_labels, event_rates),
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
    }
    return info


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

    patients_by_loc = load_patients(elec_collection, evt_collection, intraop,
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
                event_type_data_by_loc[loc_name][evt_type_name]['AUC_ROC']
                bs_info_by_loc[loc_name]['PSE'] = \
                event_type_data_by_loc[loc_name][evt_type_name][
                    'pse']  # Should agree among types, checked in db_parsing

            if filter_phfos and evt_type_name not in ['Spikes', 'Sharp Spikes']:
                patients_dic = phfo_filter(evt_type_name, patients_dic,
                                           target=filter_info['target'],
                                           tolerated_fpr=filter_info['tol_fpr'],
                                           perfect=filter_info['perfect'])
                event_type_data_by_loc[loc_name][
                    'Filtered ' + evt_type_name] = region_info(patients_dic,
                                                               [evt_type_name],
                                                               flush=True)

    graphics.event_rate_by_loc(event_type_data_by_loc, saving_path)


def compare_subtypes_rate_by_loc(elec_collection, evt_collection, hfo_type_name,
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
            elec_filter, hfo_filter = query_filters(intraop, [hfo_type_name],
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
                                           tolerated_fpr=filter_info['tol_fpr'],
                                           perfect=filter_info['perfect'])
                subtype_data_by_loc[loc_name][
                    'Filtered_' + hfo_type_name] = region_info(patients_dic,
                                                               [hfo_type_name])

    graphics.event_rate_by_loc(subtype_data_by_loc, zoomed_type=hfo_type_name,
                               saving_path=saving_path)
    plt.show()


def first_key(dic):
    return [k for k in dic.keys()][0]


if __name__ == "__main__":
    main()

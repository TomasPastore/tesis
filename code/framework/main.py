import copy
import random
import sys
from sys import version as py_version
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import \
    pyplot as plt  # todo ver donde se usa, deberia estar solo en graphics
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon, ranksums, kstest, ks_2samp
import math as mt
from datetime import timedelta
from config import (EVENT_TYPES, HFO_TYPES, exp_save_path, TEST_BEFORE_RUN,
                    EXPERIMENTS_FOLDER, experiment_default_path,
                    intraop_patients, non_intraop_patients,
                    electrodes_query_fields, hfo_query_fields, models_to_run)
from db_parsing import Database, parse_patients, get_locations, \
    get_granularity, ALL_loc_names, \
    encode_type_name, all_loc_names, load_patients, query_filters, \
    all_subtype_names
from metrics import print_metrics
from ml_hfo_classifier import phfo_filter, phfo_predictor, gather_folds, \
    get_soz_confidence_thresh, phfo_thresh_filter

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    from utils import histograms, phase_coupling_paper_polar
from utils import all_subsets, LOG, print_info, time_counter
import tests
import unittest
import graphics
import math as mt



def main():
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    # Thesis
    if TEST_BEFORE_RUN:
        # TODO Agregar a informe, tests
        unittest.main(tests, exit=False)

    run_experiment(elec_collection, evt_collection, number=4, roman_num='i',
                   letter='a')

    # TODO week 26/7 dump to overleaf reeplazing commented structure to code structure with if
    # Trabajo futuro combinacion de subtipos, mas datos,


def run_experiment(elec_collection, evt_collection, number, roman_num,
                   letter):
    NOT_IMPLEMENTED_EXP = NotImplementedError('Not implemented experiment')
    REVIEW_AND_INFORM = RuntimeError('Last review to inform')
    if number == 1:
        print('Running exp 1) Data Global analysis')
        # Total patient count, intraop vs Non intraop, other info
        # IO have artifacts, discarded ? TODO ask shennan
        # TODO correr con el nuevo info y agregar a informe
        # global_info_in_loc(elec_collection, evt_collection,
        #                     intraop=False, loc_granularity = 0,
        #                     locations = 'all', event_type_names = EVENT_TYPES)
        # show_patients_by_epilepsy_loc(soz_restricted = ['Temporal Lobe'],
        # soz_required=['Temporal Lobe'])
        raise REVIEW_AND_INFORM
    elif number == 2:
        if roman_num == 'i':
            print('Running exp 2.i) HFO rate in SOZ vs NSOZ in Whole brain')
            raise NOT_IMPLEMENTED_EXP
        elif roman_num == 'ii':
            print('Running exp 2.ii) HFO rate in SOZ vs NSOZ localized')
            raise NOT_IMPLEMENTED_EXP
        else:  # roman_num
            raise NOT_IMPLEMENTED_EXP
    elif number == 3:
        # Se quiere mejorar el rendimiento del rate de los distintos tipos
        # de HFO, para eso veamos los baselines
        # TODO move to soz_predictor module
        # TODO add to overleaf
        if roman_num == '0':
            print('Running exp 3.0) Predicting SOZ with rates: Baselines '
                  '(first steps)')
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
                                                       evt_types_to_load=HFO_TYPES + [
                                                           'Spikes'],
                                                       evt_types_to_cmp=[[t] for
                                                                         t in
                                                                         HFO_TYPES + [
                                                                             'Spikes']],
                                                       saving_path=
                                                       exp_save_path[
                                                           3]['i']['a'])
            elif letter == 'b':
                print('Running exp 3.i.b) Predicting SOZ with rates: '
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
                                                                         t in
                                                                         HFO_TYPES + [
                                                                             'Spikes']],
                                                       saving_path=
                                                       exp_save_path[
                                                           3]['i']['b'])
            else:  # letter
                raise NOT_IMPLEMENTED_EXP
        elif roman_num == 'ii':
            print('Running exp 3.ii: Ranking table of AUC baselines. '
                  'Proportion of soz electrodes AUC relation')
            pse_hfo_rate_auc_relation(elec_collection, evt_collection)

        elif roman_num == 'iii':
            print('Running exp 3.iii. Predicting SOZ with rates: '
                  'Baselines (Localized x,y,z and loc tagged)')

            def test_localized_time():
                evt_rate_soz_pred_baseline_localized(elec_collection,
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
                                                     saving_path=
                                                     exp_save_path[3]['ii'][
                                                         'dir'])

            time_counter(test_localized_time)
        else:  # roman_num
            raise NOT_IMPLEMENTED_EXP
    elif number == 4:
        from partition_builder import ml_field_names
        # 4) ML HFO classifiers para extremos de 3 iii
        if roman_num == 'i':
            if letter == 'a':
                # Whole brain without x, y, z in ml
                saving_path = str(Path(exp_save_path[4]['i']['a'],
                                       '4ia_coords_untag'))
                patients_dic, \
                baselines_data = evt_rate_soz_pred_baseline_whole_brain(
                    elec_collection,
                    evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    load_untagged_loc_from_db=True,
                    restrict_to_tagged_coords=False,
                    restrict_to_tagged_locs=False,
                    evt_types_to_load=HFO_TYPES,
                    evt_types_to_cmp=[[t] for
                                      t in
                                      HFO_TYPES],
                    saving_path= saving_path)
                feature_statistical_tests(patients_dic,
                                          types=['RonO', 'Fast RonO'],
                                          features=ml_field_names('RonO',
                                                                  include_coords=False),
                                          saving_path=saving_path)
                feature_statistical_tests(patients_dic,
                                          types=['RonS', 'Fast RonS'],
                                          features=ml_field_names('RonS',
                                                                  include_coords=False),
                                          saving_path=saving_path)
            elif letter == 'b':
                # Whole brain with x, y, z in ml
                saving_path = str(Path(exp_save_path[4]['i']['b'],
                                       '4ib_coords_tag'))
                patients_dic, \
                baselines_data = evt_rate_soz_pred_baseline_whole_brain(
                    elec_collection,
                    evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    load_untagged_loc_from_db=True,
                    restrict_to_tagged_coords=True,
                    restrict_to_tagged_locs=False,
                    evt_types_to_load=HFO_TYPES,
                    evt_types_to_cmp=[[t] for
                                      t in
                                      HFO_TYPES],
                    saving_path= saving_path)
                feature_statistical_tests(patients_dic,
                                          types=['RonO', 'Fast RonO'],
                                          features=ml_field_names('RonO',
                                                                  include_coords=True),
                                          saving_path=saving_path)

                feature_statistical_tests(patients_dic,
                                          types=['RonS', 'Fast RonS'],
                                          features=ml_field_names('RonS',
                                                                  include_coords=True),
                                          saving_path=saving_path)
        elif roman_num == 'ii':
            # Localized: Hsippocampus
            saving_path =  exp_save_path[4]['ii']['Hippocampus']
            patients_dic, \
            baselines_data = evt_rate_soz_pred_baseline_whole_brain(
                elec_collection,
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
                saving_path=saving_path)
            feature_statistical_tests(patients_dic,
                                      location='Hippocampus',
                                      types=['RonO', 'Fast RonO'],
                                      features=ml_field_names('RonO',
                                                              include_coords=True),
                                      saving_path=saving_path)
            feature_statistical_tests(patients_dic,
                                      location='Hippocampus',
                                      types=['RonS', 'Fast RonS'],
                                      features=ml_field_names('RonS',
                                                              include_coords=True),
                                      saving_path=saving_path)
        # Usar sklearn pipeline
        # Resultados: xgboost, model_patients (%75), random partition, robust scaler, balanced, filter 0.7 da 0.8 de AP
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
    else:  # number
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
                                    load_untagged_coords_from_db=True,
                                    load_untagged_loc_from_db=True)
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
                                    models_to_run,
                                    load_untagged_coords_from_db=True,
                                    load_untagged_loc_from_db=True)
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
# 3i
# Make the ROC baselines for whole brain.
# Returns the data for the ml and the data to plot the baselines
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
            type_group_name = 'HFOs' if type_group_name == 'RonO+RonS+Fast ' \
                                                           'RonO+Fast RonS' \
                else type_group_name
            print('\nInfo from: {n}'.format(n=type_group_name), file=file)
            event_type_data_by_loc[loc_name][type_group_name] = region_info(
                patients_dic,
                event_type_names)
            print_info(event_type_data_by_loc[loc_name][type_group_name],
                       file=file)

    graphics.event_rate_by_loc(event_type_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               roc_saving_path=saving_path)

    return patients_dic, event_type_data_by_loc


# 3 ii
# Localized solo hay un baseline que es el que tiene taggeado los
# x,y,z y el loc porque tienen que tener la loc definida
# Se hace un llamado a la db por localizacion porque no hace falta
# Doesnt care of angles null, in ml we will zoom that.
# Only for HFOs, not Spikes
def pse_hfo_rate_auc_relation(elec_collection, evt_collection):
    data_by_loc = evt_rate_soz_pred_baseline_localized(elec_collection,
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
                                                           g: ALL_loc_names(g)
                                                           for g
                                                           in [2, 3, 5]},
                                                       saving_path=
                                                       exp_save_path[3]['ii'][
                                                           'dir'])

    graphics.plot_pse_hfo_rate_auc_table(data_by_loc, str(Path(exp_save_path[3][
                                                                   'ii']['dir'],
                                                               'table')))

    graphics.plot_co_pse_auc(data_by_loc, str(Path(exp_save_path[3][
                                                       'ii']['dir'],
                                                   'scatter')))


# TODO ver la cantidad de electrodos empty para hacer combinacion de tipos
# FIXME hacer adaptativos los cuadros de las figuras para q se guarden bien
# 3.iii
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
                                         locations={g: all_loc_names(g) for g
                                                    in [2, 3, 5]},
                                         saving_path=
                                         exp_save_path[3]['iii']['dir']):
    print('SOZ predictor localized')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        str(load_untagged_coords_from_db)))
    print(
        'load_untagged_loc_from_db: {0}'.format(str(load_untagged_loc_from_db)))
    print(
        'restrict_to_tagged_coords: {0}'.format(str(restrict_to_tagged_coords)))
    print('restrict_to_tagged_locs: {0}'.format(str(restrict_to_tagged_locs)))
    print(elec_collection.distinct('loc2'))
    print(elec_collection.distinct('loc3'))
    print(elec_collection.distinct('loc5'))
    print(evt_collection.distinct('loc2'))
    print(evt_collection.distinct('loc3'))
    print(evt_collection.distinct('loc5'))

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
            file_saving_path = str(Path(saving_path,
                                        'loc_{g}'.format(g=granularity),
                                        loc_name.replace(' ', '_'),
                                        '3_iii_' +
                                        loc_name.replace(' ', '_') +
                                        '_sleep_tagged'))
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
                                           location=loc_name if local_filter else None)
                    min_pat_count_in_location = 12
                    min_pat_with_epilepsy_in_location = 3
                    if loc_info['patient_count'] >= min_pat_count_in_location \
                            and loc_info['patients_with_epilepsy'] >= 3:
                        print('Files_saving_path {0}'.format(file_saving_path))

                        if loc_name not in data_by_loc.keys():
                            data_by_loc[loc_name] = dict()
                        event_type_data_by_loc[loc_name][type_group_name] = \
                            loc_info
                        print_info(loc_info, file=file)

                        # Data for 3.iii table
                        data_by_loc[loc_name]['PSE'] = loc_info['pse']
                        data_by_loc[loc_name][type_group_name + '_AUC'] = \
                            loc_info['AUC_ROC']
                    else:
                        print('Region and type excluded because lack of data '
                              '--> {'
                              '0} {1}'.format(loc_name, type_group_name))
        '''
        graphics.event_rate_by_loc(event_type_data_by_loc,
                                   metrics=['pse', 'pnee', 'auc'],
                                   roc_saving_path=str(Path(saving_path,
                                    'loc_{g}'.format(g=granularity),
                                    '3_iii_sleep_tagged')),
                                   change_tab_path=True)
        '''
    return data_by_loc


# 5) ML HFO classifiers
# Compare modelos con y sin balanceo, el scaler, la forma de hacer la particion de pacientes y param tuning
# Model patients da peor

def feature_statistical_tests(patients_dic,
                              location=None,
                              types=HFO_TYPES,
                              features=['duration', 'freq_pk', 'power_pk'],
                              saving_path=exp_save_path[4]['dir']):
    # Structure initialization
    feature_data = dict()
    stats = dict()
    for feature in features:
        if 'angle' in feature:
            feature_data['sin_' + feature] = dict()
            feature_data['cos_' + feature] = dict()
            stats['sin_' + feature] = dict()
            stats['cos_' + feature] = dict()
        else:
            feature_data[feature] = dict()
            stats[feature] = dict()

        for t in types:
            if 'angle' in feature:
                feature_data['sin_'+feature][t] = {'soz': [], 'nsoz': []}
                feature_data['cos_'+feature][t] = {'soz': [], 'nsoz': []}
            else:
                feature_data[feature][t] = {'soz': [], 'nsoz': []}

    granularity = get_granularity(location)
    # Gathering data
    for p in patients_dic.values():
        if location is None:
            electrodes = p.electrodes
        else:
            electrodes = [e for e in p.electrodes if
                          getattr(e, 'loc{g}'.format(g=
                          get_granularity(
                              location)))]
        for e in electrodes:
            for t in types:
                for h in e.events[t]:
                    for f in features:
                        soz_label = 'soz' if h.info['soz'] else 'nsoz'
                        if 'angle' in f:
                            if h.info[f] == True:
                                feature_data['sin_{f}'.format(f=f)][
                                    t][soz_label].append(mt.sin(h.info[f]))
                                feature_data['cos_{f}'.format(f=f)][
                                    t][soz_label].append(mt.cos(h.info[f]))
                        else:
                            feature_data[f][t][soz_label].append(h.info[f])

    # Calculating Stat and pvalue and plotting
    for f in features:
        if 'angle' in f:
            f_names = ['sin_{f}'.format(f=f), 'cos_{f}'.format(f=f)]
        else:
            f_names = [f]
        for t in types:
            for feat_name in f_names:
                print('Feature name ', feat_name)
                print('Type ', t)

                if min(len(feature_data[feat_name][t]['soz']),
                       len(feature_data[feat_name][t]['nsoz'])) == 0:
                    print('There is no info for {f} with type {'
                      't}'.format(f=feat_name, t=t))
                else:
                    stats[feat_name][t] = dict()
                    stats[feat_name][t]['stat'], stats[feat_name][t]['pval'] \
                        = \
                        ks_2samp(feature_data[
                                     feat_name][
                                     t]['soz'],
                                 feature_data[feat_name][
                                     t][
                                     'nsoz']
                                 )
                    graphics.plot_feature_distribution(feature_data[feat_name][t][
                                                'soz'],
                                      feature_data[feat_name][t]['nsoz'],
                                      feature=feat_name,
                                      type=t,
                                      stat=stats[feat_name][t]['stat'],
                                      pval=format(stats[feat_name][t]['pval'], '.2e'),
                                      saving_path=saving_path)

# default ml algorithms comparisons in region
def ml_phfo_models(elec_collection, evt_collection, rm_null_coords, location,
                   models_to_run=models_to_run, comp_with='',
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
    hfo_cursor = evt_collection.find(evt_filter,
                                     projection=hfo_query_fields)
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
                                                  hfo_type_name,
                                                  thresh=thresh,
                                                  perfect=False,
                                                  model_name=model_name)
            confidence = None if model_name != 'Simulated' else conf
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
                               conf=conf)


# 6) pHFOs rate VS HFO rate baseline
# Tambien probe en vez de usar la prop de phfos > thresh en vez de hfo rate
# only added to test a classifier calc+ug baseline rate as feature
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
        ml_phfo_models(elec_collection, evt_collection, 'Hippocampus',
                       'RonS',
                       tol_fprs=tol_fprs, models_to_run=models_to_run,
                       comp_with=comp_with, conf=conf)


###################             Auxiliary functions             #########################################

# Gathers info about patients rate data for the types included in the list
# If loc is None all the dic is considered, otherwise only the location asked
def region_info(patients_dic, event_types=EVENT_TYPES, flush=False,
                conf=None, location=None):
    # print('Region info location {0}, types {1}.'.format(location, event_types))
    patients_with_epilepsy = set()
    elec_count_per_patient = []
    elec_x_null, elec_y_null, elec_z_null = 0, 0, 0  # todo create dic
    elec_cnt_loc2_empty, elec_cnt_loc3_empty, elec_cnt_loc5_empty = 0, 0, 0  # todo create dic
    pat_with_x_null, pat_with_y_null, pat_with_z_null = set(), set(), set()  # todo create dic
    pat_with_loc2_empty, pat_with_loc3_empty, pat_with_loc5_empty = set(), set(), set()  # todo create dic
    soz_elec_count, elec_with_evt_count, event_count = 0, 0, 0
    counts = {type: 0 for type in event_types}
    event_rates, soz_labels = [], []

    if location is not None:
        patients_dic = {p_name: p for p_name, p in patients_dic.items() if \
                        p.has_elec_in(loc=location)}

    for p_name, p in patients_dic.items():
        if location is None:
            electrodes = p.electrodes
        else:
            electrodes = [e for e in p.electrodes if getattr(e,
                                                             'loc{'
                                                             'i}'.format(i=
                                                             get_granularity(
                                                                 location)))
                          == location]

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
                for fileblock, count in e.evt_count[type].items():
                    counts[type] += count

            event_rates.append(
                e.get_events_rate(event_types))  # Measured in events/min
            soz_labels.append(e.soz)

    elec_count = sum(elec_count_per_patient)
    pse = soz_elec_count / elec_count  # proportion of soz electrodes
    non_empty_elec_prop = elec_with_evt_count / elec_count
    try:
        auc_roc = roc_auc_score(soz_labels, event_rates)
    except ValueError:
        auc_roc = None
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


def first_key(dic):
    return [k for k in dic.keys()][0]


if __name__ == "__main__":
    main()

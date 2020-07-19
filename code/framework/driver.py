from sys import version as py_version
import warnings
from pathlib import Path

import graphics
import scratch
from soz_predictor import evt_rate_soz_pred_baseline_whole_brain, \
    pse_hfo_rate_auc_relation, evt_rate_soz_pred_baseline_localized
from stats import feature_statistical_tests, hfo_rate_statistical_tests, \
    build_stat_table

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import (EVENT_TYPES, HFO_TYPES, exp_save_path)
from db_parsing import all_loc_names, ALL_loc_names
from ml_hfo_classifier import ml_hfo_classifier

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
from utils import time_counter


class Driver():
    def __init__(self, elec_collection, evt_collection):
        self.elec_collection = elec_collection
        self.evt_collection = evt_collection

    def run_experiment(self, number, roman_num=None, letter=None):
        NOT_IMPLEMENTED_EXP = NotImplementedError('Not implemented experiment')
        if number == 1:
            print('\nRunning exp 1) Data Global analysis')

            scratch.print_non_intraop_patients(self.elec_collection,
                                               self.evt_collection)

            # Get global data such as event count, elec count, in whole brain
            scratch.global_info_in_locations(self.elec_collection,
                                             self.evt_collection,
                                             intraop=False,
                                             locations={0: ['Whole Brain']},
                                             event_type_names=HFO_TYPES + [
                                                 'Spikes'],
                                             restrict_to_tagged_coords=False,
                                             restrict_to_tagged_locs=False,
                                             saving_path=str(
                                                 Path(exp_save_path[1],
                                                      'whole_brain_table')))
            scratch.global_info_in_locations(self.elec_collection,
                                             self.evt_collection,
                                             intraop=False,
                                             locations={
                                                 g: ALL_loc_names(g)
                                                 for g
                                                 in [2, 3, 5]},
                                             event_type_names=HFO_TYPES + [
                                                 'Spikes'],
                                             restrict_to_tagged_coords=True,
                                             restrict_to_tagged_locs=True,
                                             saving_path=str(
                                                 Path(exp_save_path[1],
                                                      'localized_table')))
        elif number == 2:
            if roman_num == 'i':
                print('\nRunning exp 2.i) HFO rate in SOZ vs NSOZ in Whole brain')
                location = 'Whole Brain'
                event_type_data_by_loc, \
                data_by_loc = evt_rate_soz_pred_baseline_localized(
                    self.elec_collection,
                    self.evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    load_untagged_loc_from_db=True,
                    restrict_to_tagged_coords=False,
                    restrict_to_tagged_locs=False,
                    evt_types_to_load=HFO_TYPES,
                    evt_types_to_cmp=[[t] for
                                      t in
                                      HFO_TYPES],
                    locations={0: [location]},
                    saving_dir=exp_save_path[2]['i']
                )
                stats = dict()
                stats[location] = hfo_rate_statistical_tests(
                    rates_by_type={t: data_by_loc[location][t + '_rates']
                                   for t in HFO_TYPES},
                    types=HFO_TYPES,
                    saving_dir=exp_save_path[2]['i']
                )

                # Stats tables
                columns, rows = build_stat_table(locations = [location],
                                 feat_name='HFO_rate',
                                 stats=stats)
                graphics.plot_score_in_loc_table(columns, rows,
                        colors=None,
                        saving_path=str(Path(exp_save_path[2]['i'],
                                             'stats_table'))
                )
            elif roman_num == 'ii':
                print('\nRunning exp 2.ii) HFO rate in SOZ vs NSOZ localized')
                def ba(id):
                    return 'Brodmann area {id}'.format(id=id)

                # Defined in experiment 1 by being the regions with more data
                # and more SOZ patients.
                priority_locs = {
                    2: ['Temporal Lobe', 'Limbic Lobe', 'Frontal Lobe'],
                    3: ['Parahippocampal Gyrus',
                        'Middle Temporal Gyrus', 'Sub-Gyral',
                        'Superior Temporal Gyrus', 'Uncus',
                        'Fusiform Gyrus'],
                    5: ['Hippocampus', 'Amygdala', ba(20),
                        ba(21), ba(28), ba(36)]
                    }
                event_type_data_by_loc, \
                data_by_loc = evt_rate_soz_pred_baseline_localized(
                    self.elec_collection,
                    self.evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    load_untagged_loc_from_db=True,
                    restrict_to_tagged_coords=True,
                    restrict_to_tagged_locs=True,
                    evt_types_to_load=HFO_TYPES,
                    evt_types_to_cmp=[[t] for
                                      t in
                                      HFO_TYPES],
                    locations=priority_locs,
                    saving_dir=exp_save_path[2]['ii']
                )
                stats = dict()
                locations = [loc for locs in priority_locs.values() for loc in \
                        locs]
                for loc, locs in priority_locs.items():
                    for loc_name in locs:
                        stats[loc_name] = hfo_rate_statistical_tests(
                            rates_by_type={t: data_by_loc[loc_name][t + '_rates']
                                           for t in HFO_TYPES},
                            types=HFO_TYPES,
                            saving_dir=str(Path(exp_save_path[2]['ii'],
                                                 'loc_{g}'.format(g=loc),
                                                loc_name.replace(' ', '_')))
                        )
                # Stats tables
                columns, rows = build_stat_table(
                    locations=locations,
                    feat_name='HFO_rate',
                    stats=stats)

                graphics.plot_score_in_loc_table(columns, rows,
                                                 colors=None,
                                                 saving_path=str(Path(
                                                     exp_save_path[2][
                                                         'ii'],
                                                     'stats_table'))
                                                 )
            else:  # roman_num
                raise NOT_IMPLEMENTED_EXP
        elif number == 3:
            # Se quiere mejorar el rendimiento del rate de los distintos tipos
            # de HFO, para eso veamos los baselines
            # TODO add to overleaf
            # TODO change for localized version
            if roman_num == '0':
                print('\nRunning exp 3.0) Predicting SOZ with rates: Baselines '
                      '(first steps)')
                evt_rate_soz_pred_baseline_whole_brain(self.elec_collection,
                                                       self.evt_collection,
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
                    print('\nRunning exp 3.i.a) Predicting SOZ with rates: '
                          'Baselines (Whole brain coords untagged)')
                    # TODO change for localized version

                    # Ma)ML with 91 patients without using coords (untagged)
                    # Whole brain rates for independent event types:
                    evt_rate_soz_pred_baseline_whole_brain(self.elec_collection,
                                                           self.evt_collection,
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
                    print('\nRunning exp 3.i.b) Predicting SOZ with rates: '
                          'Baselines (Whole brain coords tagged)')
                    # Mb) 57 patients with tagged coords.
                    # NOTE: V0_ if you just dont load untagged coords from db,
                    # it improves AUC 2%, we loose 34 patients, but there are
                    # still 2 electrodes in None because of bad format of
                    # the field (empty lists map to None) from db
                    # TODO change for localized version

                    evt_rate_soz_pred_baseline_whole_brain(self.elec_collection,
                                                           self.evt_collection,
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
                print('\nRunning exp 3.ii: Ranking table of AUC baselines. '
                      'Proportion of soz electrodes AUC relation')
                pse_hfo_rate_auc_relation(self.elec_collection,
                                          self.evt_collection)
            elif roman_num == 'iii':
                print('\nRunning exp 3.iii. Predicting SOZ with rates: '
                      'Baselines (Localized x,y,z and loc tagged)')
                def test_localized_time():
                    evt_rate_soz_pred_baseline_localized(self.elec_collection,
                                                         self.evt_collection,
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
                                                         locations={
                                                             g: all_loc_names(g)
                                                             for g
                                                             in [2, 3, 5]},
                                                         saving_dir=
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
                    print('\nRunning exp 4.i.a) ML HFO classifier for Whole '
                          'Brain with coords untagged')
                    saving_dir = exp_save_path[4]['i']['a']  # dir
                    saving_path = str(Path(saving_dir, 'untagged'))
                    baselines_data, patients_dic = evt_rate_soz_pred_baseline_whole_brain(
                        self.elec_collection,
                        self.evt_collection,
                        intraop=False,
                        load_untagged_coords_from_db=True,
                        load_untagged_loc_from_db=True,
                        restrict_to_tagged_coords=False,
                        restrict_to_tagged_locs=False,
                        evt_types_to_load=HFO_TYPES,
                        evt_types_to_cmp=[[t] for
                                          t in
                                          HFO_TYPES],
                        saving_path=saving_path)
                    feature_statistical_tests(patients_dic,
                                              types=['RonO', 'Fast RonO'],
                                              features=ml_field_names('RonO',
                                                                      include_coords=False),
                                              saving_dir=saving_dir)
                    feature_statistical_tests(patients_dic,
                                              types=['RonS', 'Fast RonS'],
                                              features=ml_field_names('RonS',
                                                                      include_coords=False),
                                              saving_dir=saving_dir)
                    for hfo_type in HFO_TYPES:
                        ml_hfo_classifier(patients_dic,
                                          location='Whole Brain',
                                          hfo_type=hfo_type,
                                          use_coords=False,
                                          ml_models=['XGBoost'],
                                          saving_dir=str(Path(saving_dir,
                                                              hfo_type)))

                elif letter == 'b':
                    print('\nRunning exp 4.i.b) ML HFO classifier for Whole '
                          'Brain with coords tagged')
                    # Whole brain with x, y, z in ml
                    saving_dir = exp_save_path[4]['i']['b']  # dir
                    saving_path = str(Path(saving_dir, 'tagged'))

                    baselines_data, patients_dic = evt_rate_soz_pred_baseline_whole_brain(
                        self.elec_collection,
                        self.evt_collection,
                        intraop=False,
                        load_untagged_coords_from_db=True,
                        load_untagged_loc_from_db=True,
                        restrict_to_tagged_coords=True,
                        restrict_to_tagged_locs=False,
                        evt_types_to_load=HFO_TYPES,
                        evt_types_to_cmp=[[t] for
                                          t in
                                          HFO_TYPES],
                        saving_path=saving_path)
                    feature_statistical_tests(patients_dic,
                                              types=['RonO', 'Fast RonO'],
                                              features=ml_field_names('RonO',
                                                                      include_coords=True),
                                              saving_dir=saving_dir)

                    feature_statistical_tests(patients_dic,
                                              types=['RonS', 'Fast RonS'],
                                              features=ml_field_names('RonS',
                                                                      include_coords=True),
                                              saving_dir=saving_dir)
                    for hfo_type in HFO_TYPES:
                        ml_hfo_classifier(patients_dic,
                                          location='Whole Brain',
                                          hfo_type=hfo_type,
                                          use_coords=True,
                                          saving_dir=str(Path(saving_dir,
                                                              hfo_type)))
            elif roman_num == 'ii':
                if letter == 'a':
                    print('\nRunning exp 4.ii a) ML HFO classifier in '
                          'Hippocampus. Tagged but without using x, y, '
                          'z in ml because of epilepsy localization bias.')
                    # Localized: Hippocampus
                    saving_dir = exp_save_path[4]['ii']['Hippocampus']  # dir
                    baselines_data, data_by_loc = \
                        evt_rate_soz_pred_baseline_localized(self.elec_collection,
                                                             self.evt_collection,
                                                             intraop=False,
                                                             load_untagged_coords_from_db=True,
                                                             load_untagged_loc_from_db=True,
                                                             restrict_to_tagged_coords=True,
                                                             restrict_to_tagged_locs=True,
                                                             evt_types_to_load=HFO_TYPES,
                                                             evt_types_to_cmp=[[t]
                                                                               for
                                                                               t in
                                                                               HFO_TYPES],
                                                             locations={
                                                                 5: 'Hippocampus'},
                                                             saving_dir=saving_dir,
                                                             loc_pat_dic='Hippocampus')
                    patients_dic = data_by_loc['Hippocampus']['patients_dic']
                    print('Hippocampus pat dic {0}'.format(patients_dic))
                    feature_statistical_tests(patients_dic,
                                              location='Hippocampus',
                                              types=['RonO', 'Fast RonO'],
                                              features=ml_field_names('RonO',
                                                                      include_coords=True),
                                              saving_dir=saving_dir)
                    feature_statistical_tests(patients_dic,
                                              location='Hippocampus',
                                              types=['RonS', 'Fast RonS'],
                                              features=ml_field_names('RonS',
                                                                      include_coords=True),
                                              saving_dir=saving_dir)
                    for hfo_type in HFO_TYPES:
                        ml_hfo_classifier(patients_dic,
                                          location='Hippocampus',
                                          hfo_type=hfo_type,
                                          use_coords=False,
                                          saving_dir=str(Path(saving_dir,
                                                              hfo_type)))
            # Usar sklearn pipeline
            # Resultados: xgboost, model_patients (%75), random partition, robust scaler, balanced, filter 0.7 da 0.8 de AP
            # hippocampus_hfo_classifier(self.elec_collection,
            # self.evt_collection) #
        elif number == 5:
            # 5) pHFOs rate VS HFO rate baseline
            # TODO week 12/7
            raise NOT_IMPLEMENTED_EXP
            # phfo_rate_vs_baseline_whole_brain(self.elec_collection,
            # self.evt_collection,
            # allow_null_coords=True, event_type_names) #TODO
            # phfo_rate_vs_baseline_whole_brain(self.elec_collection,
            # self.evt_collection,
            # allow_null_coords=True) #TODO

            # compare_Hippocampal_RonS_ml_models(self.elec_collection,
            # self.evt_collection)
            # Hippocampal_RonS_gradual_filters(self.elec_collection,
            # self.evt_collection)
            # Hippocampal_RonS_ml_with_rate(self.elec_collection,
            # self.evt_collection) #
        # TODO
        #  mencionar en discusion de resultados de la comparacion de baseline vs ml filters
        elif number == 6:
            # TODO week 19/7
            raise NOT_IMPLEMENTED_EXP
            # 6) Simulation of the ml predictor to understand needed performance to
            # improve HFO rate baseline
            # simulator(self.elec_collection, self.evt_collection) TODO
            #  mencionar en
            #  discusion
            #  de
            #  resultados de la comparacion de baseline vs ml filters
        else:  # number
          raise NOT_IMPLEMENTED_EXP


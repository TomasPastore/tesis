from sys import version as py_version
import warnings
from pathlib import Path

import graphics
import scratch
from soz_predictor import pse_hfo_rate_auc_relation, evt_rate_soz_pred_baseline_localized
from stats import feature_statistical_tests, hfo_rate_statistical_tests, \
    build_stat_table
warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import (EVENT_TYPES, HFO_TYPES, exp_save_path)
from db_parsing import preference_locs, all_loc_names
from ml_hfo_classifier import ml_hfo_classifier, \
    ml_hfo_classifier_sk_learn_train

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
from partition_builder import ml_field_names
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
                                                 g: all_loc_names(g)
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
                print('\nRunning data analysis for Whole Brain')
                if letter == 'a':
                    print('Coords Untagged allowed')
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    locations = {0: ['Whole Brain']}
                    saving_dir = exp_save_path[2]['i']['a']
                elif letter =='b':
                    print('Coords Tagged requiered')
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = False
                    locations = {0: ['Whole Brain']}
                    saving_dir = exp_save_path[2]['i']['b']
                else:
                    raise NOT_IMPLEMENTED_EXP
            elif roman_num == 'ii':
                print('\nRunning data analysis localized')
                restrict_to_tagged_coords = True
                restrict_to_tagged_locs = True
                locations = {g: preference_locs(g) for g in [2, 3, 5]}
                saving_dir = exp_save_path[2]['ii']
            else:  # roman_num
                raise NOT_IMPLEMENTED_EXP

            event_type_data_by_loc, \
            data_by_loc = evt_rate_soz_pred_baseline_localized(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                load_untagged_coords_from_db=True,
                load_untagged_loc_from_db=True,
                restrict_to_tagged_coords=restrict_to_tagged_coords,
                restrict_to_tagged_locs=restrict_to_tagged_locs,
                evt_types_to_load=HFO_TYPES,
                evt_types_to_cmp=[[t] for
                                  t in
                                  HFO_TYPES],
                locations=locations,
                saving_dir=saving_dir,
                return_pat_dic_by_loc=True
            )

            print('\nFeatures and HFO rate distributions SOZ vs NSOZ')

            location_names = [loc for locs in locations.values() for loc in \
                         locs]
            # Esto no cambia con lo de kmeans, quizas algun path
            stats = dict()
            for loc, locs in locations:
                for location in locs:
                    saving_dir_feat = saving_dir if location == 'Whole ' \
                                                              'Brain' \
                        else str(Path(saving_dir,
                                'loc_{g}'.format(g=loc),
                                location.replace(' ', '_')))

                    # Features
                    # TODO retornar stats con features para tabla
                    patients_dic = data_by_loc[location]['patients_dic']
                    feature_statistical_tests(patients_dic,
                                              location=location,
                                              types=['RonO', 'Fast RonO'],
                                              features=ml_field_names('RonO',
                                                       include_coords=True),
                                              saving_dir=saving_dir_feat)
                    feature_statistical_tests(patients_dic,
                                              location=location,
                                              types=['RonS', 'Fast RonS'],
                                              features=ml_field_names('RonS',
                                                  include_coords=True),
                                              saving_dir=saving_dir_feat)
                    # HFO rate
                    stats[location] = hfo_rate_statistical_tests(
                        rates_by_type={t: data_by_loc[location][t + '_rates']
                                       for t in HFO_TYPES},
                        types=HFO_TYPES,
                        saving_dir=saving_dir_feat
                    )

            # Stat tables
            for feature in set(['HFO rate']+
                               ml_field_names('RonO')+
                               ml_field_names('RonS')):
                #TODO for every feature
                if feature == 'HFO rate':
                    columns, rows = build_stat_table(
                        locations=location_names,
                        feat_name=feature,
                        stats=stats)

                    graphics.plot_score_in_loc_table(columns, rows,
                                                     colors=None,
                                                     saving_path=str(Path(
                                                         saving_dir,
                                                         '{f}_stats_table'.format(f=feature)))
                                                     )
        elif number == 3:
            # Se quiere mejorar el rendimiento del rate de los distintos tipos
            # de HFO, para eso veamos los baselines
            # TODO add to overleaf
            if roman_num == '0':
                print('\nRunning exp 3.0) Predicting SOZ with rates: Baselines '
                      '(first steps)')
                restrict_to_tagged_coords = False
                restrict_to_tagged_locs = False
                evt_types_to_load = EVENT_TYPES
                evt_types_to_cmp = [HFO_TYPES, ['Spikes']]
                locations = {0: ['Whole Brain']}
                saving_dir = exp_save_path[3]['i']['0']


            elif roman_num == 'i':
                evt_types_to_load = HFO_TYPES + ['Spikes']
                evt_types_to_cmp = [[t] for t in HFO_TYPES + ['Spikes']]
                if letter == 'a':
                    print('\nRunning exp 3.i.a) Predicting SOZ with rates: '
                          'Baselines (Whole brain coords untagged)')
                    # Ma)ML with 91 patients without using coords (untagged)
                    # Whole brain rates for independent event types
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    locations = {0: ['Whole Brain']}
                    saving_dir = exp_save_path[3]['i']['a']

                elif letter == 'b':
                    print('\nRunning exp 3.i.b) Predicting SOZ with rates: '
                          'Baselines (Whole brain coords tagged)')
                    # Mb) 57 patients with tagged coords.
                    # NOTE: V0_ if you just dont load untagged coords from db,
                    # it improves AUC 2%, we loose 34 patients, but there are
                    # still 2 electrodes in None because of bad format of
                    # the field (empty lists map to None) from db
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations = {0: ['Whole Brain']}
                    saving_dir = exp_save_path[3]['i']['b']

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
                restrict_to_tagged_coords = True
                restrict_to_tagged_locs = True
                evt_types_to_load = HFO_TYPES + ['Spikes']
                evt_types_to_cmp = [[t] for t in HFO_TYPES + ['Spikes']]
                locations = {g: preference_locs(g) for g in [2, 3, 5]}
                saving_dir = exp_save_path[3]['iii']['dir']
            else:  # roman_num
                raise NOT_IMPLEMENTED_EXP

            if roman_num in ['0','i','iii']:
                evt_rate_soz_pred_baseline_localized(self.elec_collection,
                                                     self.evt_collection,
                                                     intraop=False,
                                                     load_untagged_coords_from_db=True,
                                                     load_untagged_loc_from_db=True,
                                                     restrict_to_tagged_coords=restrict_to_tagged_coords,
                                                     restrict_to_tagged_locs=restrict_to_tagged_locs,
                                                     evt_types_to_load=evt_types_to_load,
                                                     evt_types_to_cmp=evt_types_to_cmp,
                                                     locations=locations,
                                                     saving_dir=saving_dir,
                                                     plot_rocs=True)
        elif number == 4:
            # 4) ML HFO classifiers para extremos de 3 iii
            evt_types_to_load= HFO_TYPES #Cargo los 4 por consistencia y para
            # tener todos los baseline
            if roman_num == 'i':
                if letter == 'a': #DEPRECADO ?
                    print('\nRunning exp 4.i.a) ML HFO classifier for Whole '
                          'Brain with coords untagged')
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    locations = locations = {0: ['Whole Brain']}
                    saving_dir = exp_save_path[4]['i']['a']  # dir
                    use_coords = False

                elif letter == 'b':
                    print('\nRunning exp 4.i.b) ML HFO classifier for Whole '
                          'Brain with coords tagged')
                    # Whole brain with x, y, z in ml
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations =  locations= {0: ['Whole Brain']}
                    saving_dir = exp_save_path[4]['i']['b']  # dir
                    use_coords = True #TODO ver si se usa
                else:
                    raise NOT_IMPLEMENTED_EXP
            elif roman_num == 'ii':
                if letter == 'a':
                    print('\nRunning exp 4.ii a) ML HFO classifier in '
                          'Hippocampus. Tagged but without using x, y, '
                          'z in ml because of epilepsy localization bias.')
                    # Localized: Hippocampus
                    location= 'Hippocampus'
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations = {5: [location]}
                    saving_dir = exp_save_path[4]['ii'][location]  # dir
                    use_coords = False
                else: # Letter
                    raise NOT_IMPLEMENTED_EXP
            else: # Roman num
                raise NOT_IMPLEMENTED_EXP

            baselines_data, data_by_loc = \
                evt_rate_soz_pred_baseline_localized(
                    self.elec_collection,
                    self.evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    load_untagged_loc_from_db=True,
                    restrict_to_tagged_coords=restrict_to_tagged_coords,
                    restrict_to_tagged_locs=restrict_to_tagged_locs,
                    evt_types_to_load=evt_types_to_load, #HFO_TYPES
                    evt_types_to_cmp=[[t] for t in evt_types_to_load],
                    locations=locations,
                    saving_dir=saving_dir,
                    return_pat_dic_by_loc=True,
                    plot_rocs=False)

            for locs in locations.values():
                for location in locs:
                    patients_dic = data_by_loc[location]['patients_dic'] #ya
                    # filtrado el diccionario de pacientes en loc
                    for hfo_type in evt_types_to_load:
                        if hfo_type in ['RonO']:
                            continue
                        ml_hfo_classifier_sk_learn_train(patients_dic,
                                                         location=location,
                                                         #solo para saber en
                                                         # que location
                                                         # estoy, pero el
                                                         # pat_dic
                                                         # ya esta filtrado
                                                         hfo_type=hfo_type,
                                                         use_coords=use_coords,
                                                         saving_dir=saving_dir)
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


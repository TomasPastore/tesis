import warnings
from pathlib import Path

import graphics
import scratch
from soz_predictor import pse_hfo_rate_auc_relation, \
    evt_rate_soz_pred_baseline_localized
from stats import feature_statistical_tests, hfo_rate_statistical_tests, \
    build_stat_table

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from conf import FIG_SAVE_PATH
from db_parsing import EVENT_TYPES, HFO_TYPES, preference_locs, \
    all_loc_names, WHOLE_BRAIN_L0C
from ml_hfo_classifier import ml_hfo_classifier, \
    ml_hfo_classifier_sk_learn_train

from partition_builder import ml_field_names


class Driver():
    def __init__(self, elec_collection, evt_collection):
        self.elec_collection = elec_collection
        self.evt_collection = evt_collection

    def run_experiment(self, number, roman_num=None, letter=None):
        NOT_IMPLEMENTED_EXP = NotImplementedError('Not implemented experiment')
        if number == 1:
            print('\nRunning exp 1) Data Global analysis...')
            print('Sleep patients and data dimensions...')

            scratch.print_non_intraop_patients(self.elec_collection,
                                               self.evt_collection)

            # TODO mention untagged data and discard untagged to simplify
            # TODO update manuscript tables with True True
            scratch.global_info_in_locations(self.elec_collection,
                                             self.evt_collection,
                                             intraop=False,
                                             locations={0: [WHOLE_BRAIN_L0C]},
                                             event_type_names=HFO_TYPES + [
                                                 'Spikes'],
                                             restrict_to_tagged_coords=False,
                                             # TODO update with True
                                             restrict_to_tagged_locs=False,
                                             # TODO idem above
                                             saving_path=str(
                                                 Path(FIG_SAVE_PATH[1],
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
                                                 Path(FIG_SAVE_PATH[1],
                                                      'localized_table')))
        elif number == 2:
            print('\nRunning exp 2) Data stats analysis...')
            print('Features and HFO rate distributions SOZ vs NSOZ')
            if roman_num == 'i':
                if letter == 'a':
                    print('2.i.a Whole Brain coordinates untagged')
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[2]['i']['a']
                elif letter == 'b':  # TODO rerun after artifact correction
                    print('2.i.a Whole Brain, coords and locs tagged required')
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[2]['i']['b']
                else:
                    raise NOT_IMPLEMENTED_EXP
            elif roman_num == 'ii':
                print('2.ii Localized, coords and locs tagged')
                restrict_to_tagged_coords = True
                restrict_to_tagged_locs = True
                locations = {g: preference_locs(g) for g in [2, 3, 5]}
                saving_dir = FIG_SAVE_PATH[2]['ii']
            else:  # roman_num
                raise NOT_IMPLEMENTED_EXP

            data_by_loc = evt_rate_soz_pred_baseline_localized(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                load_untagged_coords_from_db=True,  # to solve inconsistencies
                load_untagged_loc_from_db=True,  # to solve inconsistencies
                restrict_to_tagged_coords=restrict_to_tagged_coords,  # In RAM
                restrict_to_tagged_locs=restrict_to_tagged_locs,  # In RAM
                evt_types_to_load=HFO_TYPES,
                evt_types_to_cmp=[[t] for t in HFO_TYPES],
                locations=locations,
                saving_dir=saving_dir,
                return_pat_dic_by_loc=True,  # This is data for later doing ml
                remove_elec_artifacts=True  # TODO TESTING
            )

            location_names = [l for locs in locations.values() for l in locs]
            stats = dict()  # Saves stats by location for table generation
            for loc, locs in locations.items():
                for location in locs:
                    if location == WHOLE_BRAIN_L0C:
                        saving_dir_feat = saving_dir
                    else:
                        saving_dir_feat = str(Path(saving_dir,
                                                   'loc_{g}'.format(g=loc),
                                                   location.replace(' ', '_')))

                    # Features
                    # TODO return stats features for table
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

            # Generate stat tables TODO for features, just hfo rate for now
            features = set(['HFO_rate'] +
                           ml_field_names('RonO', include_coords=True) +
                           ml_field_names('RonS', include_coords=True))
            for feature in features:
                # TODO for every feature
                if feature == 'HFO_rate':
                    columns, rows = build_stat_table(
                        locations=location_names,
                        feat_name=feature,
                        stats=stats)

                    graphics.plot_score_in_loc_table(columns, rows,
                                                     colors=None,
                                                     saving_path=str(Path(
                                                         saving_dir,
                                                         '{f}_stats_table'.format(
                                                             f=feature)))
                                                     )
        elif number == 3:
            print('\nRunning exp 3) Event rate soz prediction baselines...')
            evt_types_to_load = HFO_TYPES + ['Spikes']  # todo ask if sharp
            # spikes are worthy to include in baselines
            if roman_num == '0':
                # TODO update manuscript with tagged data
                print('Running exp 3.0) simple model all HFOs vs Spikes')
                restrict_to_tagged_coords = True  # It was False first
                restrict_to_tagged_locs = True  # It was False first
                evt_types_to_cmp = [HFO_TYPES, ['Spikes']]
                locations = {0: [WHOLE_BRAIN_L0C]}
                saving_dir = FIG_SAVE_PATH[3]['i']['0']

            elif roman_num == 'i':
                if letter == 'a':
                    print('Running exp 3.i.a) Whole Brain untagged (N = 91)')
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    evt_types_to_cmp = [[t] for t in HFO_TYPES + ['Spikes']]
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[3]['i']['a']

                elif letter == 'b':
                    print('Running exp 3.i.b) Whole Brain tagged (N = 57)')
                    # NOTE: then if you just dont load untagged coords from db,
                    # it improves AUC 2%, we loose 34 patients
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    evt_types_to_cmp = [[t] for t in HFO_TYPES + ['Spikes']]
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[3]['i']['b']

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
                saving_dir = FIG_SAVE_PATH[3]['iii']['dir']
            else:  # roman_num
                raise NOT_IMPLEMENTED_EXP

            if roman_num in ['0', 'i', 'iii']:
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
            print('\nRunning exp 4) Machine learning HFO classifiers...')
            evt_types_to_load = HFO_TYPES
            if roman_num == 'i':
                if letter == 'a':
                    print('Running exp 4.i.a) Whole Brain untagged')
                    restrict_to_tagged_coords = False
                    restrict_to_tagged_locs = False
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[4]['i']['a']  # dir
                    use_coords = False

                elif letter == 'b':
                    print('Running exp 4.i.b) Whole Brain tagged')
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations = {0: [WHOLE_BRAIN_L0C]}
                    saving_dir = FIG_SAVE_PATH[4]['i']['b']  # dir
                    use_coords = False  # TODO test using True for whole brain
                else:
                    raise NOT_IMPLEMENTED_EXP

            elif roman_num == 'ii':
                if letter == 'a':
                    print('Running exp 4.ii.a Localized Hippocampus')
                    location = 'Hippocampus'
                    restrict_to_tagged_coords = True
                    restrict_to_tagged_locs = True
                    locations = {5: [location]}
                    saving_dir = FIG_SAVE_PATH[4]['ii'][location]  # dir
                    use_coords = False
                else:  # Letter
                    raise NOT_IMPLEMENTED_EXP
            else:  # Roman num
                raise NOT_IMPLEMENTED_EXP

            data_by_loc = evt_rate_soz_pred_baseline_localized(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                load_untagged_coords_from_db=True,
                load_untagged_loc_from_db=True,
                restrict_to_tagged_coords=restrict_to_tagged_coords,
                restrict_to_tagged_locs=restrict_to_tagged_locs,
                evt_types_to_load=evt_types_to_load,
                evt_types_to_cmp=[[t] for t in evt_types_to_load],
                locations=locations,
                saving_dir=saving_dir,
                return_pat_dic_by_loc=True,
                plot_rocs=False)

            for locs in locations.values():
                for location in locs:
                    patients_dic = data_by_loc[location]['patients_dic']
                    print('Patients in {0}: {1}'.format(location, patients_dic))
                    for hfo_type in evt_types_to_load:
                        # TODO remove temporal condition just to run Fast RonO
                        if hfo_type != 'Fast RonO':
                            continue
                        # Here below the location parameter is just for the
                        # string because the data dictionary has already been
                        # filtered
                        ml_hfo_classifier_sk_learn_train(patients_dic,
                                                         location=location,
                                                         hfo_type=hfo_type,
                                                         use_coords=use_coords,
                                                         saving_dir=saving_dir)
        elif number == 5:
            # 5) pHFOs rate VS HFO rate baseline
            # TODO
            raise NOT_IMPLEMENTED_EXP
            # REVIEW OLD CODE BELOW
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

        elif number == 6:
            # TODO
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

import warnings
from pathlib import Path
from sys import version as py_version

from graphics import plot_global_info_by_loc_table
from soz_predictor import region_info, first_key

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import (EVENT_TYPES, intraop_patients, models_to_run,
                    non_intraop_patients, exp_save_path)
from db_parsing import get_locations, \
    load_patients

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
from utils import all_subsets


# 1) Data Global analysis  #####################################################################
# Electrodes collection didn't have intraop field, we got all the patients from all the events
def print_non_intraop_patients(electrodes_collection, hfo_collection):
    print('Looking for non intraop patients...')
    elec_cursor = electrodes_collection.find({}, projection=['patient_id'])
    intraop_hfo_cursor = hfo_collection.find({'intraop': '1'},
                                             projection=['patient_id',
                                                         'intraop'])
    non_intraop_hfo_cursor = hfo_collection.find({'intraop': '0'},
                                                 projection=['patient_id',
                                                             'intraop'])

    elec_patients = set()
    for e in elec_cursor:
        elec_patients.add(e['patient_id'])

    intraop_hfo_patients = set()
    for h in intraop_hfo_cursor:
        intraop_hfo_patients.add(h['patient_id'])

    non_intraop_hfo_patients = set()
    for h in non_intraop_hfo_cursor:
        non_intraop_hfo_patients.add(h['patient_id'])

    print('Uncertain patients (appear in intraop and non intraop)')
    in_both = intraop_hfo_patients.intersection(non_intraop_hfo_patients)
    print(in_both)

    hfo_tot_patients = intraop_hfo_patients.union(non_intraop_hfo_patients)
    print('Total patient count: {0}'.format(len(elec_patients)))
    print('Patient list: {0}'.format(sorted(list(elec_patients)) ) )

    assert(hfo_tot_patients==elec_patients)

    assert(set(non_intraop_patients) == non_intraop_hfo_patients-in_both)
    print('Non intraop count: {0}'.format(len(non_intraop_patients)))
    print('Non intraop Patient list: {0}'.format(non_intraop_patients) )


def global_info_in_locations(elec_collection, evt_collection, intraop=False,
                             locations={0:['Whole Brain']},
                             event_type_names=EVENT_TYPES,
                             restrict_to_tagged_coords=False,
                             restrict_to_tagged_locs=False,
                             saving_path= str(Path(exp_save_path[1], 'table'))):
    print('Gathering global info...')
    patients_by_loc = load_patients(elec_collection, evt_collection,
                                    intraop,
                                    loc_granularity=0,
                                    locations=['Whole Brain'],
                                    event_type_names=event_type_names,
                                    models_to_run=models_to_run,
                                    load_untagged_coords_from_db=True,
                                    load_untagged_loc_from_db=True,
                                    restrict_to_tagged_coords=restrict_to_tagged_coords,
                                    restrict_to_tagged_locs=restrict_to_tagged_locs)
    data_by_loc = dict()
    whole_brain_name = first_key(patients_by_loc)
    patients_dic = patients_by_loc[whole_brain_name]

    for loc_names in locations.values():
        for loc_name in loc_names:
            loc_info = region_info(patients_dic,
                                   event_types=event_type_names,
                                   location=loc_name if loc_name !=whole_brain_name
                                                     else None)
            min_pat_count_in_location = 12
            min_pat_with_epilepsy_in_location = 3
            if loc_info['patient_count'] >= min_pat_count_in_location \
                    and loc_info[
                'patients_with_epilepsy'] >= min_pat_with_epilepsy_in_location:

                data_by_loc[loc_name] = dict()
                data_by_loc[loc_name]['patient_count'] = loc_info[
                    'patient_count']
                data_by_loc[loc_name]['patients_with_epilepsy'] = loc_info[
                    'patients_with_epilepsy']
                data_by_loc[loc_name]['elec_count'] = loc_info[
                    'elec_count']
                data_by_loc[loc_name]['soz_elec_count'] = loc_info[
                    'soz_elec_count']
                data_by_loc[loc_name]['PSE'] = loc_info['pse']
                for t in event_type_names:
                    data_by_loc[loc_name][t+'_N'] = loc_info['evt_count_per_type'][t]

                #print('Global info in location: {0} \n {1}'.format(loc_name,
                # data_by_loc[loc_name]))

    plot_global_info_by_loc_table(data_by_loc, saving_path)
import copy
import math as mt

import numpy as np
from astropy import units as u
from astropy.stats import rayleightest
from scipy.stats import circmean

HFO_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']

def save_json(electrodes_info_p):
    import json
    with open("patient_electrode_info.json", "w") as file:
        json.dump(electrodes_info_p, file, indent=4, sort_keys=True)

def encode_type_name(name):
    return str(HFO_TYPES.index(name) + 1)

inconsistencies = {}
def log(msg=None, msg_type=None, patient=None, electrode=None):
    print_msg = False
    if print_msg and msg is not None:
        print(msg)

    if msg_type is not None:
        assert (patient is not None)
        if not patient in inconsistencies.keys():
            inconsistencies[patient] = dict()

        if msg_type == 'BLOCK_DURATION':
            if not msg_type in inconsistencies[patient].keys():
                inconsistencies[patient][msg_type] = 0

            inconsistencies[patient][msg_type] +=1
        else:
            assert (electrode is not None)

            if not electrode in inconsistencies[patient].keys():
                inconsistencies[patient][electrode] = dict()

            if not msg_type in inconsistencies[patient][electrode].keys():
                inconsistencies[patient][electrode][msg_type] = 0

            inconsistencies[patient][electrode][msg_type] += 1


def unique_patients(collection, crit):
    cursor = collection.find(crit)
    docs = set()
    for doc in cursor:
        docs.add(doc['patient_id'])
    patient_ids = list(docs)
    patient_ids.sort()
    print("Unique patients count: {0}".format(len(patient_ids)))
    print(patient_ids)
    return patient_ids


def electrode_pat_vs_hfo_pat(electrodes_collection, hfo_collection, loc_filter, hfo_types):
    loc_name = [v for v in loc_filter.values()]
    print('Patients with electrodes in {0}:'.format(loc_name))
    with_electrodes = unique_patients(electrodes_collection, loc_filter)
    with_electrodes.sort()
    only_electrodes = dict()
    only_hfos = dict()
    for hfo_type_name in hfo_types:
        hfo_filter = copy.deepcopy(loc_filter)
        hfo_filter['intraop'] = '0'
        hfo_filter['type'] = encode_type_name(hfo_type_name)
        print('Patients with hfos in {0}, intraop==0 and type {1}'.format(loc_name, hfo_type_name))
        with_hfos= unique_patients(hfo_collection, hfo_filter)
        with_hfos.sort()
        only_electrodes[hfo_type_name] = set(with_electrodes) - set(with_hfos)
        only_hfos[hfo_type_name] = set(with_hfos) - set(with_electrodes)

    return only_electrodes, only_hfos

def angle_clusters(collection, hfo_filter, angle_field_name, amp_step):
    angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
    docs = collection.find(hfo_filter)
    hfo_count = docs.count()
    angles = []
    for doc in docs:
        angle = doc[angle_field_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
        angles.append(angle)
        angle_group_id = mt.floor(angle / amp_step)
        angle_grouped[str(angle_group_id)] += 1  # increment count of group

    for k, v in angle_grouped.items():  # normalizing values
        r_value = round((v / hfo_count) * 100, 2)  # show them as relative percentages
        angle_grouped[k] = r_value

    mean_angle = mt.degrees(circmean(angles))
    pvalue = float(rayleightest(np.array(angles) * u.rad))  # doctest: +FLOAT_CMP
    hfo_count_2 = sum([v for v in angle_grouped.values()]) #Todo extract
    assert(hfo_count == hfo_count)
    return angle_grouped, mean_angle, pvalue, hfo_count


import math as mt

import numpy as np
from astropy import units as u
from astropy.stats import rayleightest
from scipy.stats import circmean

inconsistencies = {}

def save_json(electrodes_info_p):
    import json
    with open("patient_electrode_info.json", "w") as file:
        json.dump(electrodes_info_p, file, indent=4, sort_keys=True)


def log(msg=None, msg_type=None, patient=None, electrode=None):
    print_msg = False
    if print_msg and msg is not None:
        print(msg)

    if msg_type is not None:
        assert (patient is not None)
        assert (electrode is not None)

        if not patient in inconsistencies.keys():
            inconsistencies[patient] = dict()

        if not electrode in inconsistencies[patient].keys():
            inconsistencies[patient][electrode] = dict()

        if not msg_type in inconsistencies[patient][electrode].keys():
            inconsistencies[patient][electrode][msg_type] = 0

        inconsistencies[patient][electrode][msg_type] += 1


# Fast queries
def unique_patients(collection, crit):
    # Unique patients ids for filter crit
    # Usage example
    # unique_crit = {'$and': [{'intraop': '0'}, {'loc5': 'Brodmann area 21'}]}
    # unique_patients(hfo_collection, unique_crit)
    hfos_in_zone = collection.find(crit)
    docs = set()
    for doc in hfos_in_zone:
        docs.add(doc['patient_id'])
    patient_ids = list(docs)
    patient_ids.sort()
    print("Unique patients count: {0}".format(len(patient_ids)))
    print(patient_ids)
    return patient_ids


#TODO update
def angle_clusters(collection, crit, angle_name,  amp_step=np.pi/9):
    angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
    docs = collection.find(crit)
    hfo_count = docs.count()
    angles = []
    for doc in docs:
        angle = doc[angle_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
        angles.append(angle)
        angle_group_id = mt.floor(angle / amp_step)
        angle_grouped[str(angle_group_id)] += 1  # increment count of group

    for k, v in angle_grouped.items():  # normalizing values
        r_value = round((v / hfo_count) * 100, 2)  # show them as relative percentages
        angle_grouped[k] = r_value

    mean_angle = mt.degrees(circmean(angles))
    pvalue = float(rayleightest(np.array(angles) * u.rad))  # doctest: +FLOAT_CMP

    return angle_grouped, amp_step, mean_angle, pvalue, hfo_count

import json
import random
import time
from itertools import chain, combinations
from pathlib import Path

import numpy as np


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def save_json(info_dic, saving_path):
    with open(saving_path, "w") as file:
        json.dump(info_dic, file, indent=4, sort_keys=True)


def load_json(saving_path):
    with open(saving_path) as json_file:
        return json.load(json_file)


LOG = {}


def log(msg=None, msg_type=None, patient=None, electrode=None):
    print_msg = False
    if print_msg and msg is not None:
        print(msg)

    if msg_type is not None:
        assert (patient is not None)
        if not patient in LOG.keys():
            LOG[patient] = dict()

        if msg_type == 'BLOCK_DURATION':
            if not msg_type in LOG[patient].keys():
                LOG[patient][msg_type] = 0

            LOG[patient][msg_type] += 1
        else:
            assert (electrode is not None)

            if not electrode in LOG[patient].keys():
                LOG[patient][electrode] = dict()

            if not msg_type in LOG[patient][electrode].keys():
                LOG[patient][electrode][msg_type] = 0

            LOG[patient][electrode][msg_type] += 1


def time_counter(callable_code):
    start_time = time.clock()
    res = callable_code()
    print('Runned in {0} minutes'.format(round((time.clock() -
                                                start_time) / 60), 2))
    return res


def constant(f):
    def fset(self, value):
        raise TypeError('You cant modify this constant')

    def fget(self):
        return f()

    return property(fget, fset)


@constant
def FOO():
    return 'constant'


# Randomly maps patient id strings to 1... N
def map_pat_ids(model_patients):
    patient_names = [p.id for p in model_patients]
    random.shuffle(patient_names)
    return [patient_names.index(p.id) + 1 for p in model_patients]


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


def print_info(info_evts, file):
    # Printing format for saving miscellaneous data in .txt files
    attributes_width = max(len(key) for key in info_evts.keys()) + 2  # padding
    header = '{attr} || {value}'.format(
        attr='Attributes'.ljust(attributes_width), value='Values')
    sep = ''.join(['-' for i in range(len(header))])
    print(header, file=file)
    print(sep, file=file)
    for k, v in info_evts.items():
        k, v = (k, v) if k != 'evt_rates' else ('mean_of_rates', np.mean(v))
        if isinstance(v, list) or k == 'patients_dic_in_loc':  # too long to be
            # printed
            continue
        else:
            row = '{attr} || {value}'.format(attr=k.ljust(attributes_width),
                                             value=str(v))
            print(row, file=file)

    pat_coord_null = info_evts['pat_with_x_null'].union(
        info_evts['pat_with_y_null'])
    pat_coord_null = pat_coord_null.union(info_evts['pat_with_z_null'])
    print('Count of patients with null coords: {0}'.format(len(pat_coord_null)),
          file=file)
    print('List of patients with null coords: {0}'.format(pat_coord_null),
          file=file)

    pat_with_empty_loc = info_evts['pat_with_loc2_empty'].union(
        info_evts['pat_with_loc3_empty'])
    pat_with_empty_loc = pat_with_empty_loc.union(
        info_evts['pat_with_loc5_empty'])
    print(
        'Count of patients with empty loc: {0}'.format(len(pat_with_empty_loc)),
        file=file)
    print('List of patients with empty loc: {0}'.format(pat_with_empty_loc),
          file=file)


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1)))

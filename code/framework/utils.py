import copy
import math as mt

import numpy as np
from astropy import units as u
from astropy.stats import rayleightest, kuiper
from scipy.stats import circmean, circstd

from config import intraop_patients
import graphics
import matlab.engine

EVENT_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']

def encode_type_name(name):
    return str(EVENT_TYPES.index(name) + 1)


def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name

def save_json(electrodes_info_p):
    import json
    with open("patient_electrode_info.json", "w") as file:
        json.dump(electrodes_info_p, file, indent=4, sort_keys=True)


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

            inconsistencies[patient][msg_type] += 1
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


def all_subtype_names(hfo_type_name):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        subtypes = ['slow', 'delta', 'theta', 'spindle']
    else:
        subtypes = ['spike']
    return subtypes


def get_granularity(loc):
    for i in range(6):
        if loc in all_loc_names(i):
            return i
    raise RuntimeError('Unknown location name')


def all_loc_names(granularity):
    if granularity == 0:
        return ['Whole brain']
    elif granularity == 1:
        return ['Right Cerebrum', 'Left Cerebrum']
    elif granularity == 2:
        return ['Limbic Lobe', 'Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe', 'Occipital Lobe']  #
    elif granularity == 3:
        return ['Fusiform Gyrus', 'Parahippocampal Gyrus',
                'Middle Temporal Gyrus', 'Postcentral Gyrus',
                'Superior Frontal Gyrus', 'Inferior Frontal Gyrus', 'Middle Frontal Gyrus']
    elif granularity == 4:
        return ['Gray Matter', 'White Matter']
    elif granularity == 5:
        return ['Hippocampus', 'Amygdala', 'Brodmann area 21', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37']


# PAPER PHASE COUPLING AUXS

def angle_clusters(collection, hfo_filter, angle_field_name, amp_step):
    angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
    docs = collection.find(hfo_filter)
    if hfo_filter['soz'] == '0':
        main_soz = False
        hfo_filter_2 = copy.deepcopy(hfo_filter)
        hfo_filter_2['soz'] = '1'
    else:
        main_soz = True
        hfo_filter_2 = copy.deepcopy(hfo_filter)
        hfo_filter_2['soz'] = '0'
    docs2 = collection.find(hfo_filter_2)
    hfo_count = docs.count()
    angles = []
    angles2 = []
    pat_elec = dict()
    for doc in docs:
        angle = doc[angle_field_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
        angles.append(angle)
        angle_group_id = mt.floor(angle / amp_step)
        angle_grouped[str(angle_group_id)] += 1  # increment count of group

        pat_id = doc['patient_id']
        e_name = parse_elec_name(doc)
        if pat_id not in pat_elec.keys():
            pat_elec[pat_id] = set()
        pat_elec[pat_id].add(e_name)

    for k, v in angle_grouped.items():  # normalizing values
        r_value = round((v / hfo_count) * 100, 2)  # show them as relative percentages
        angle_grouped[k] = r_value

    for doc in docs2:
        angle = doc[angle_field_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
        angles2.append(angle)

    elec_count = sum([len(elec_set) for elec_set in pat_elec.values()])
    circ_mean = mt.degrees(circmean(angles))
    circ_std = mt.degrees(circstd(angles))

    pvalue = float(rayleightest(np.array(angles) * u.rad))  # doctest: +FLOAT_CMP
    # D, fpp = kuiper(np.array(angles))  # doctest: +FLOAT_CMP
    matlab_engine = get_matlab_session()
    alpha1 = angles2 if main_soz else angles  # nsoz angles
    alpha2 = angles if main_soz else angles2  # soz angles
    matlab_engine.addpath('/home/tpastore/Documentos/lic_computacion/tesis/code/scratch/matlab_circ_package')
    print(len(alpha1))
    print(len(alpha2))

    alpha1 = matlab.double(alpha1)
    alpha2 = matlab.double(alpha2)
    k_pval, k, K = matlab_engine.circ_kuipertest(alpha2, alpha1, nargout=3)
    print(k_pval)
    print(k)
    print(K)
    # D, fpp = 'TODO', 'TODO'  # doctest: +FLOAT_CMP

    return angle_grouped, circ_mean, circ_std, pvalue, k_pval, hfo_count, elec_count


def parse_soz(db_representation_str):
    return (True if db_representation_str == "1" else False)


def histograms(hfo_collection):
    hfos = hfo_collection.find(filter={'patient_id': {'$nin': intraop_patients},
                                       'type': encode_type_name('RonO'),
                                       'slow': 1,
                                       'intraop': '0',
                                       'loc5': 'Hippocampus'},
                               projection=['soz', 'slow_angle', 'power_pk', 'freq_pk', 'duration'])

    feature_names = ['Duration', 'Spectral content', 'Power Peak', 'Slow angle']
    data = {f_name: dict(soz=[], n_soz=[]) for f_name in feature_names}

    for h in hfos:
        power_pk = float(mt.log10((h['power_pk'])))
        freq_pk = float(h['freq_pk'])
        duration = float(h['duration']) * 1000  # seconds to milliseconds
        slow_angle = float(h['slow_angle']) / np.pi

        soz = parse_soz(h['soz'])
        soz_key = 'soz' if soz else 'n_soz'

        data['Power Peak'][soz_key].append(power_pk)
        data['Spectral content'][soz_key].append(freq_pk)
        data['Duration'][soz_key].append(duration)
        data['Slow angle'][soz_key].append(slow_angle)

    # NORMAL HIST
    # for feature_name in ['Duration', 'Spectral content', 'Power Peak']:
    #    graphics.hist_feature_distributions(feature_name, data[feature_name]['soz'], data[feature_name]['n_soz'])

    # 2D HIST / HEAT MAP
    graphics.hist2d_feature_distributions('Duration', data)
    graphics.hist2d_feature_distributions('Power Peak', data)
    graphics.hist2d_feature_distributions('Spectral content', data)

    # BOX PLOT
    # graphics.boxplot_feature_distributions('Duration', data)
    # graphics.boxplot_feature_distributions('Power Peak', data)
    # graphics.boxplot_feature_distributions('Spectral content', data)


def get_matlab_session():
    return matlab.engine.start_matlab()


def phase_coupling_paper_polar(hfo_collection):
    # Phase coupling paper
    graphics.compare_hfo_angle_distribution(hfo_collection, 'RonO', 'slow',
                                            fig_title='RonO slow angle distribution in Hippocampus')
    graphics.compare_hfo_angle_distribution(hfo_collection, 'Fast RonO', 'slow',
                                            fig_title='Fast RonO slow angle distribution in Hippocampus')
    histograms(hfo_collection)  # 2d, 3d

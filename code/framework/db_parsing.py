import copy

from classes import Patient, Electrode, HFO
from config import HFO_TYPES, models_to_run
from utils import log
import math as mt


def parse_patients(electrodes_cursor, hfo_cursor):
    patients_dic = dict()
    parse_electrodes(patients_dic, electrodes_cursor)
    parse_hfos(patients_dic, hfo_cursor)
    add_empty_blocks(patients_dic)

    return patients_dic


def add_empty_blocks(patients_dic):
    # Just in case, if a patient has blocks 1 and 3 for electrode e, I add 2 because that block was missing
    empty_blocks_added = 0
    for p in patients_dic.values():
        # Add block id for hfo_rate
        for i in range(1, max(p.file_blocks.keys())):
            file_block = i
            if file_block not in p.file_blocks.keys():
                p.file_blocks[file_block] = None
                empty_blocks_added += 1

    print('Empty blocks added: {0}'.format(empty_blocks_added))


def parse_age(doc):
    return 0.0 if doc['age'] == "empty" else float(doc['age'])


def parse_soz(db_representation_str):
    return (True if db_representation_str == "1" else False)


def decode_type_name(type_id):
    return HFO_TYPES[int(type_id) - 1]


def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name


def parse_loc(doc, i):
    if isinstance(doc['loc{0}'.format(i)], str) and len(doc['loc{0}'.format(i)]) > 0:
        loc = doc['loc{0}'.format(i)]
    elif isinstance(doc['loc{0}'.format(i)], list) and len(doc['loc{0}'.format(i)]) > 0:
        if isinstance(doc['loc{0}'.format(i)][0], str) and len(doc['loc{0}'.format(i)][0]) > 0:
            loc = doc['loc{0}'.format(i)][0]
        elif isinstance(doc['loc{0}'.format(i)][0], list) and len(doc['loc{0}'.format(i)][0]) > 0:
            assert (isinstance(doc['loc{0}'.format(i)][0][0], str) and len(doc['loc{0}'.format(i)][0][0]) > 0)
            loc = doc['loc{0}'.format(i)][0][0]
        else:
            loc = 'empty'

    else:
        loc = 'empty'

    assert (isinstance(loc, str))
    return loc


def parse_electrodes(patients, elec_cursor):
    for e in elec_cursor:
        # Patient level
        patient_id = e['patient_id']
        file_block = int(e['file_block'])
        if not patient_id in patients.keys():
            patient = Patient(id=patient_id, age=parse_age(e), file_blocks={file_block: None})
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(e)
            if age != patient.age:
                log('Warning, age should agree among electrodes of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)
            if file_block not in patient.file_blocks.keys():
                patient.file_blocks[file_block] = None

        # Electrode level
        e_name = parse_elec_name(e)
        loc1 = parse_loc(e, 1)
        loc2 = parse_loc(e, 2)
        loc3 = parse_loc(e, 3)
        loc4 = parse_loc(e, 4)
        loc5 = parse_loc(e, 5)

        if not e_name in patient.electrode_names():
            electrode = Electrode(e_name, parse_soz(e['soz']), parse_soz(e['soz_sc']),
                                  loc1=loc1, loc2=loc2, loc3=loc3, loc4=loc4, loc5=loc5)
            patient.add_electrode(electrode)
        else:
            electrode = next(e2 for e2 in patient.electrodes if e2.name == e_name)
            # Check consistency
            if parse_soz(e['soz']) != electrode.soz or parse_soz(e['soz_sc']) != electrode.soz_sc:
                log(msg=('Warning, soz disagreement among blocks of '
                         'the same patient_id, electrode, '
                         'running OR between values'),
                    msg_type='SOZ_electrode',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                electrode.soz = electrode.soz or parse_soz(e['soz'])
                electrode.soz_sc = electrode.soz_sc or parse_soz(e['soz_sc'])

            ##Locations
            if electrode.loc1 == 'empty':
                electrode.loc1 = loc1
            if electrode.loc2 == 'empty':
                electrode.loc2 = loc2
            if electrode.loc3 == 'empty':
                electrode.loc3 = loc3
            if electrode.loc4 == 'empty':
                electrode.loc4 = loc4

            if electrode.loc5 == 'empty':
                electrode.loc5 = loc5
            elif loc5 != 'empty' and electrode.loc5 != loc5:
                log(msg=('Warning, loc5 disagreement among blocks for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':  # Priority
                    electrode.loc5 = loc5


def parse_hfos(patients, hfo_collection):
    for h in hfo_collection:
        # Patient level
        patient_id = h['patient_id']
        file_block = int(h['file_block'])
        block_duration = float(h['r_duration'])
        if patient_id not in patients.keys():
            patient = Patient(id=patient_id, age=parse_age(h), file_blocks={file_block: block_duration})
            patients[patient.id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(h)
            if age != patient.age:
                log('Warning, age should be consistent among hfos of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

            if file_block not in patient.file_blocks.keys() or patient.file_blocks[file_block] is None:
                patient.file_blocks[file_block] = block_duration
            else:
                if block_duration != patient.file_blocks[file_block]:
                    log('Warning, block duration disagreement among hfos of the same patient, taking average',
                        msg_type='BLOCK_DURATION',
                        patient=patient.id)
                    assert(False)
                    patient.file_blocks[file_block] = (patient.file_blocks[file_block] + block_duration) / 2

        # Electrode level
        e_name = parse_elec_name(h)
        loc1 = parse_loc(h, 1)
        loc2 = parse_loc(h, 2)
        loc3 = parse_loc(h, 3)
        loc4 = parse_loc(h, 4)
        loc5 = parse_loc(h, 5)

        if not e_name in patient.electrode_names():
            electrode = Electrode(e_name, parse_soz(h['soz']), parse_soz(h['soz_sc']), loc5=loc5)
            patient.add_electrode(electrode)
        else:
            electrode = next(e for e in patient.electrodes if e.name == e_name)
            # Check consistency
            if (parse_soz(h['soz']) != electrode.soz or parse_soz(h['soz_sc']) != electrode.soz_sc):
                electrode.soz = electrode.soz or parse_soz(h['soz'])
                electrode.soz_sc = electrode.soz_sc or parse_soz(h['soz_sc'])

            ##Locations
            if electrode.loc1 == 'empty':
                electrode.loc1 = loc1
            if electrode.loc2 == 'empty':
                electrode.loc2 = loc2
            if electrode.loc3 == 'empty':
                electrode.loc3 = loc3
            if electrode.loc4 == 'empty':
                electrode.loc4 = loc4
            if electrode.loc5 == 'empty':
                electrode.loc5 = loc5
            elif loc5 != 'empty' and electrode.loc5 != loc5:
                log(msg=('Warning, loc5 disagreement among hfos for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':
                    electrode.loc5 = loc5

        # HFO_level
        info = dict(
            prediction={m: [0, 0] for m in models_to_run},  # for saving results of a model
            proba={m: 0 for m in models_to_run},  # for saving results of a model
            soz=parse_soz(h['soz']),
            type=decode_type_name(h['type']),
            file_block=int(h['file_block']),
            start_t=float(h['start_t']),
            finish_t=float(h['start_t']),
            duration=float(h['duration']) * 1000,
            fr_duration=float(h['fr_duration']),
            r_duration=float(h['r_duration']),
            freq_av=float(h['freq_av']),
            freq_pk=float(h['freq_pk']),
            power_av=mt.log10(float(h['power_av'])) if float(h['power_av']) != 0 else 0.0,
            power_pk=mt.log10(float(h['power_pk'])) if float(h['power_pk']) != 0 else 0.0,
            loc1=electrode.loc1,
            loc2=electrode.loc2,
            loc3=electrode.loc3,
            loc4=electrode.loc4,
            loc5=electrode.loc5,
            age=patient.age
        )
        if decode_type_name(h['type']) in ['RonO', 'Fast RonO']:
            info['slow'] = bool(h['slow'])
            info['slow_vs'] = 0.0 if (isinstance(h['slow_vs'], list) or h['slow_vs'] is None) else float(h['slow_vs'])
            info['slow_angle'] = 0.0 if (isinstance(h['slow_angle'], list) or h['slow_angle'] is None) else float(h['slow_angle'])
            info['delta'] = bool(h['delta'])
            info['delta_vs'] = 0.0 if (isinstance(h['delta_vs'], list) or h['delta_vs'] is None) else float(h['delta_vs'])
            info['delta_angle'] = 0.0 if (isinstance(h['delta_angle'], list) or h['delta_angle'] is None) else float(h['delta_angle'])
            info['theta'] = bool(h['theta'])
            info['theta_vs'] = 0.0 if (isinstance(h['theta_vs'], list) or h['theta_vs'] is None) else float(h['theta_vs'])
            info['theta_angle'] = 0.0 if (isinstance(h['theta_angle'], list) or h['theta_angle'] is None) else float(h['theta_angle'])
            info['spindle'] = bool(h['spindle'])
            info['spindle_vs'] = 0.0 if (isinstance(h['spindle_vs'], list) or h['spindle_vs'] is None) else float(h['spindle_vs'])
            info['spindle_angle'] = 0.0 if (isinstance(h['spindle_angle'], list) or h['spindle_angle'] is None) else float(h['spindle_angle'])
        else:
            info['spike'] = bool(h['spike'])
            info['spike_vs'] = 0.0 if (isinstance(h['spike_vs'], list) or h['spike_vs'] is None) else float(h['spike_vs'])
            info['spike_angle'] = 0.0 if (isinstance(h['spike_angle'], list) or h['spike_angle'] is None) else float(h['spike_angle'])

        hfo = HFO(info)
        electrode.add(hfo)

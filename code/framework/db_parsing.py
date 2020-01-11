

def parse_patients(electrodes_cursor, hfo_cursor):
    patients_dic = dict()
    parse_electrodes(patients_dic, electrodes_cursor)
    parse_events(patients_dic, hfo_cursor)

    return patients_dic

def parse_age(doc):
    return 0.0 if doc['age'] == "empty" else float(doc['age'])

def parse_soz(db_representation_str):
    return (True if db_representation_str == "1" else False)

def encode_type_name(name):
    return str(EVENT_TYPES.index(name) + 1)

def decode_type_name(type_id):
    return EVENT_TYPES[int(type_id) - 1]

def parse_coord(param):
    return -1.0 if isinstance(param, list) else float(param)

def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name

from utils import log, all_loc_names
from classes import Patient, Electrode, Event
from config import EVENT_TYPES, models_to_run
import math as mt

def parse_locations(loc_granularity, locations):
    if loc_granularity == 0:
        loc = None
        locations = ['Whole brain']
    else:
        loc = 'loc{i}'.format(i=loc_granularity)
        locations = all_loc_names(granularity=loc_granularity) if locations == 'all' else locations
    return loc, locations


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
            patient = Patient(id=patient_id, age=parse_age(e))
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(e)
            if age != patient.age:
                log('Warning, age should agree among electrodes of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)


        # Electrode level
        e_name = parse_elec_name(e)
        loc1 = parse_loc(e, 1)
        loc2 = parse_loc(e, 2)
        loc3 = parse_loc(e, 3)
        loc4 = parse_loc(e, 4)
        loc5 = parse_loc(e, 5)

        if not e_name in patient.electrode_names():
            electrode = Electrode(e_name, parse_soz(e['soz']), {file_block: None} , parse_soz(e['soz_sc']),
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

            if file_block not in electrode.blocks.keys():
                electrode.blocks[file_block] = None
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



def parse_events(patients, event_collection):
    for evt in event_collection:
        # Patient level
        patient_id = evt['patient_id']
        file_block = int(evt['file_block'])
        block_duration = float(evt['r_duration'])
        if patient_id not in patients.keys():
            patient = Patient(id=patient_id, age=parse_age(evt))
            patients[patient.id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(evt)
            if age != patient.age:
                log('Warning, age should be consistent among events of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

        # Electrode level
        e_name = parse_elec_name(evt)
        soz =parse_soz(evt['soz'])
        loc1 = parse_loc(evt, 1)
        loc2 = parse_loc(evt, 2)
        loc3 = parse_loc(evt, 3)
        loc4 = parse_loc(evt, 4)
        loc5 = parse_loc(evt, 5)

        if not e_name in patient.electrode_names():
            electrode = Electrode(e_name, soz, {file_block: block_duration}, parse_soz(evt['soz_sc']), loc5=loc5)
            patient.add_electrode(electrode)
        else:
            electrode = next(e for e in patient.electrodes if e.name == e_name)
            # Check consistency
            if not electrode.soz and soz:
                assert(False)

            #if (soz != electrode.soz or parse_soz(evt['soz_sc']) != electrode.soz_sc):
            #    electrode.soz = electrode.soz or soz
            #    electrode.soz_sc = electrode.soz_sc or parse_soz(evt['soz_sc'])

            if file_block not in electrode.blocks.keys() or electrode.blocks[file_block] is None:
                electrode.blocks[file_block] = block_duration
            else:
                assert(block_duration == electrode.blocks[file_block])

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
                log(msg=('Warning, loc5 disagreement among events for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':
                    electrode.loc5 = loc5
        #Elec count update
        evt_type = decode_type_name(evt['type'])
        if file_block not in electrode.evt_count[evt_type].keys():
            electrode.evt_count[evt_type][file_block] = 1
        else:
            electrode.evt_count[evt_type][file_block] += 1

        if soz:
            electrode.pevt_count[evt_type] += 1

        # HFO_level
        info = dict(
            x=parse_coord(evt['x']),
            y=parse_coord(evt['y']),
            z=parse_coord(evt['z']),
            prediction={m: 0 for m in models_to_run},  # for saving results of a model
            proba={m: 0 for m in models_to_run},  # for saving results of a model
            soz=soz,
            type=decode_type_name(evt['type']),
            file_block=int(evt['file_block']),
            start_t=float(evt['start_t']),
            finish_t=float(evt['finish_t']),
            duration=float(evt['duration']) * 1000,
            fr_duration=float(evt['fr_duration']),
            r_duration=float(evt['r_duration']),
            freq_av=float(evt['freq_av']),
            freq_pk=float(evt['freq_pk']),
            power_av=mt.log10(float(evt['power_av'])) if float(evt['power_av']) != 0 else 0.0,
            power_pk=mt.log10(float(evt['power_pk'])) if float(evt['power_pk']) != 0 else 0.0,
            loc1=electrode.loc1,
            loc2=electrode.loc2,
            loc3=electrode.loc3,
            loc4=electrode.loc4,
            loc5=electrode.loc5,
            age=patient.age
        )
        if decode_type_name(evt['type']) in ['RonO', 'Fast RonO']:
            info['slow'] = bool(evt['slow'])
            info['slow_vs'] = 0.0 if (isinstance(evt['slow_vs'], list) or evt['slow_vs'] is None) else float(evt['slow_vs'])
            info['slow_angle'] = 0.0 if (isinstance(evt['slow_angle'], list) or evt['slow_angle'] is None) else float(evt['slow_angle'])
            info['delta'] = bool(evt['delta'])
            info['delta_vs'] = 0.0 if (isinstance(evt['delta_vs'], list) or evt['delta_vs'] is None) else float(evt['delta_vs'])
            info['delta_angle'] = 0.0 if (isinstance(evt['delta_angle'], list) or evt['delta_angle'] is None) else float(evt['delta_angle'])
            info['theta'] = bool(evt['theta'])
            info['theta_vs'] = 0.0 if (isinstance(evt['theta_vs'], list) or evt['theta_vs'] is None) else float(evt['theta_vs'])
            info['theta_angle'] = 0.0 if (isinstance(evt['theta_angle'], list) or evt['theta_angle'] is None) else float(evt['theta_angle'])
            info['spindle'] = bool(evt['spindle'])
            info['spindle_vs'] = 0.0 if (isinstance(evt['spindle_vs'], list) or evt['spindle_vs'] is None) else float(evt['spindle_vs'])
            info['spindle_angle'] = 0.0 if (isinstance(evt['spindle_angle'], list) or evt['spindle_angle'] is None) else float(evt['spindle_angle'])
        else:
            info['spike'] = bool(evt['spike'])
            info['spike_vs'] = 0.0 if (isinstance(evt['spike_vs'], list) or evt['spike_vs'] is None) else float(evt['spike_vs'])
            info['spike_angle'] = 0.0 if (isinstance(evt['spike_angle'], list) or evt['spike_angle'] is None) else float(evt['spike_angle'])

        event = Event(info)
        electrode.add(event)

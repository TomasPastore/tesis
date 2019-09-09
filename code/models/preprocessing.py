from classes import Patient, Electrode, HFO
from utils import log
from config import HFO_TYPES

def soz_bool(db_representation_str):
    return (True if db_representation_str == "1" else False)

def soz_float(db_representation_str):
    return (1.0 if db_representation_str == "1" else 0.0)

def encode_type_name(name):
    return str(HFO_TYPES.index(name) + 1)

def decode_type_name(type_id):
    return HFO_TYPES[int(type_id) - 1]

def type_ids():
    return {name: str(index + 1) for index, name in enumerate(HFO_TYPES)}

def get_spike_kind(type_name):
    return False if type_name in ['RonO', 'Fast RonO'] else True

def parse_electrodes(electrodes):
    patients = dict()
    for e in electrodes:
        # Patient level
        if not e['patient_id'] in patients.keys():
            patient = Patient(
                id=e['patient_id'],
                age=0.0 if e['age'] == "empty" else float(e['age']),
                file_blocks={float(e['file_block'])}
            )
            patients[e['patient_id']] = patient
        else:
            # Check patient consistency
            patient = patients[e['patient_id']]
            age = 0.0 if e['age'] == "empty" else float(e['age'])
            if age != patient.age:
                print('Warning, age should be consistent' \
                      ' between blocks of the same patient')
                if age is not None:
                    patient.age = age
            patient.file_blocks.add(float(e['file_block']))

        # Electrode level
        if isinstance(e['loc5'], str):
            loc5 = e['loc5']
        elif isinstance(e['loc5'], list):
            loc5 = None if len(e['loc5']) == 0 else e['loc5'][0]
        else:
            raise NotImplementedError('Unknown loc5 type in parse electrodes')

        assert (isinstance(e['electrode'], list))

        if not e['electrode'][0] in patient.electrode_names():
            electrode = Electrode(
                e['electrode'][0],
                soz_bool(e['soz']),
                soz_bool(e['soz_sc']),
                loc5=loc5
            )
            patients[patient.id].add_electrode(electrode)
        else:
            electrode = next(e2 for e2 in patients[e['patient_id']].electrodes if e2.name == e['electrode'][0])
            # Check consistency
            if (soz_bool(e['soz']) != electrode.soz or
                    soz_bool(e['soz_sc']) != electrode.soz_sc):
                log(
                    msg=('Warning, soz disagreement among blocks in '
                         'the same (patient_id, electrode), '
                         'running OR between values'),
                    msg_type='SOZ_0',
                    patient=e['patient_id'],
                    electrode=e['electrode'][0]
                )
                electrode.soz = electrode.soz or soz_bool(e['soz'])
                electrode.soz_sc = electrode.soz_sc or soz_bool(e['soz_sc'])

            if electrode.loc5 is None:
                electrode.loc5 = loc5
            elif loc5 is not None and loc5 != electrode.loc5:
                log(msg=('Warning, loc5 disagreement among blocks in '
                         'the same (patient_id, electrode), '
                         'Priority to Hippocampus'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':
                    electrode.loc5 = loc5
    return patients


def parse_hfos(patients, hfo_collection, spike_kind):
    amount = 0
    for h in hfo_collection:
        amount +=1
        # Patient level
        if h['patient_id'] not in patients.keys():
            patient = Patient(
                id=h['patient_id'],
                age=0.0 if h['age'] == "empty" else float(h['age']),
                file_blocks={float(h['file_block'])},
            )
            patients[patient.id] = patient
        else:
            patient = patients[h['patient_id']]
            # Check consistency of patient attributes
            age = 0.0 if h['age'] == "empty" else float(h['age'])
            if age != patient.age:
                log('Warning, age should be consistent' \
                    ' between blocks of the same patient')
                if age is not None:
                    patient.age = age
            patient.file_blocks.add(float(h['file_block']))

        # Electrode level

        loc5 = h['loc5'] if isinstance(h['loc5'], str) and \
                            len(h['loc5']) > 0 else None

        if isinstance(h['electrode'], list):
            e_name = h['electrode'][0] if len(h['electrode']) > 0 else None
        elif isinstance(h['electrode'], str):
            e_name = h['electrode'] if len(h['electrode']) > 0 else None

        if not e_name in patient.electrode_names():
            electrode = Electrode(
                e_name,
                soz_bool(h['soz']),
                soz_bool(h['soz_sc']),
                loc5=loc5
            )
            patient.add_electrode(electrode)
        else:
            electrode = next(e for e in patient.electrodes if e.name == e_name)
            # Check consistency
            if (soz_bool(h['soz']) != electrode.soz or
                soz_bool(h['soz_sc']) != electrode.soz_sc):
                log(msg=('Warning, soz disagreement among hfos in '
                         'the same patient_id, electrode, '
                         'running OR between values'),
                    msg_type='SOZ_0',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                electrode.soz = electrode.soz or soz_bool(h['soz'])
                electrode.soz_sc = electrode.soz_sc or soz_bool(h['soz_sc'])

            if electrode.loc5 is None:
                electrode.loc5 = loc5
            elif loc5 != electrode.loc5:
                log(msg=('Warning, loc5 disagreement among blocks in '
                         'the same (patient_id, electrode), '
                         'Priority to Hippocampus'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':
                    electrode.loc5 = loc5
        # HFO_level
        info = dict(type=decode_type_name(h['type']),
                    soz=soz_bool(h['soz']),
                    file_block=float(h['file_block']),
                    duration=float(h['duration']),
                    intraop=float(h['intraop']),
                    fr_duration=float(h['fr_duration']),
                    r_duration=float(h['r_duration']),
                    freq_av=float(h['freq_av']),
                    freq_pk=float(h['freq_pk']),
                    power_av=float(h['power_av']),
                    power_pk=float(h['power_pk']),
                    slow=float(h['slow']),
                    slow_vs=0.0 if isinstance(h['slow_vs'], list) \
                        else float(h['slow_vs']),
                    slow_angle=0.0 if isinstance(h['slow_angle'], list) \
                        else float(h['slow_angle']),
                    delta=float(h['delta']),
                    delta_vs=0.0 if isinstance(h['delta_vs'], list) \
                        else float(h['delta_vs']),
                    delta_angle=0.0 if isinstance(h['delta_angle'], list) \
                        else float(h['delta_angle']),
                    theta=float(h['theta']),
                    theta_vs=0.0 if isinstance(h['theta_vs'], list) \
                        else float(h['theta_vs']),
                    theta_angle=0.0 if isinstance(h['theta_angle'], list) \
                        else float(h['theta_angle']),
                    spindle=float(h['spindle']),
                    spindle_vs=0.0 if isinstance(h['spindle_vs'], list) \
                        else float(h['spindle_vs']),
                    spindle_angle=0.0 if isinstance(h['spindle_angle'], list) \
                        else float(h['spindle_angle'])
        )
        hfo = HFO(info)
        electrode.add(hfo)
    print('{0} hfos parsed.'.format(amount))
    #print('Printing patients after parsing')
    #for k, p in patients.items():
    #    p.print()
    return patients

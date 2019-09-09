import pymongo

from preprocessing import soz_bool, get_spike_kind, parse_hfos, encode_type_name
import models
from config import HFO_TYPES
from utils import inconsistencies, log, unique_patients
from classes import Database


def load_patients_with_hfo_types(hfo_collection, type_names):

    # Todo 'outcome', 'resected',
    common_attr = ['patient_id', 'age', 'file_block',
                   'electrode', 'soz', 'soz_sc', 'loc5',
                   'type', 'duration', 'intraop',
                   'fr_duration', 'r_duration',
                   'freq_av', 'freq_pk',
                   'power_av', 'power_pk']

    patients_dic = dict()
    for type in HFO_TYPES:
        if type in type_names:
            spike_kind = get_spike_kind(type)
            patients_dic = add_patients_by_hfos(patients_dic, hfo_collection, common_attr, spike_kind, hfo_type_name=type )

    patients = []
    for id, p in patients_dic.items():
        patients.append(p)

    return patients


def add_patients_by_hfos(patients_dic, hfo_collection, common_attr, spike_kind, hfo_type_name):

    if spike_kind:
        specific_attributes =  ['spike', 'spike_vs', 'spike_angle']
    else:
        specific_attributes = ['slow', 'slow_vs', 'slow_angle',
                               'delta', 'delta_vs', 'delta_angle',
                               'theta', 'theta_vs', 'theta_angle',
                               'spindle', 'spindle_vs', 'spindle_angle']

    hfo_type = encode_type_name(hfo_type_name)
    hfos_cursor = hfo_collection.find(
        filter={'type': hfo_type, 'loc5': 'Hippocampus', 'intraop':'0'},
        projection=common_attr + specific_attributes,
        #sort= [('patient_id', pymongo.ASCENDING), ('electrode', pymongo.ASCENDING)]
    )
    #Unifying types and parsing inconsistencies
    patients_dic = parse_hfos(patients_dic, hfos_cursor, spike_kind)
    #print('Debug --> After parsing hfos patients have {0}'.format(len(patients_dic.keys())))
    return patients_dic


def add_empty_blocks(patients, electrodes_collection):
    empty_blocks_added = 0
    for p in patients:
        for e in p.electrodes:
            hfo_empty_blocks = electrodes_collection.find(
                filter={'patient_id': p.id, "$or": [{'electrode': [e.name]}, {'electrode': e.name}]},
                projection=['soz', 'file_block']
            )
            for h in hfo_empty_blocks:
                # Consistency for soz
                soz = soz_bool(h['soz'])
                if (soz != e.soz):
                    log(msg=('Warning, soz disagreement among hfos in '
                             'the same patient_id, electrode, '
                             'running OR between values'),
                        msg_type='SOZ_0',
                        patient=p.id,
                        electrode=e.name
                        )
                    e.soz = e.soz or soz

                    # Add block id for hfo_rate
                    file_block = float(h['file_block'])
                    if file_block not in p.file_blocks:
                        empty_blocks_added += 1
                    p.file_blocks.add(file_block)
    print('Empty blocks added: {0}'.format(empty_blocks_added))
    return patients

def main():
    db = Database()
    connection = db.get_connection()
    db = connection.deckard_new

    electrodes_collection = db.Electrodes
    electrodes_collection.create_index([('type', pymongo.ASCENDING)], unique = False)
    electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

    hfo_collection = db.HFOs
    hfo_collection.create_index([('type', pymongo.ASCENDING)], unique = False)
    hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

    #Debug line below
    #unique_patients(hfo_collection, {'type': '1', 'intraop':'0', 'loc5': 'Hippocampus'})

    print('Loading data from database...')

    type_names_to_run = ['RonO']
    patients = load_patients_with_hfo_types(hfo_collection, type_names_to_run)
    patients = add_empty_blocks(patients, electrodes_collection)
    print('Found some inconsistencies while parsing: {0}'.format(inconsistencies))

    total_hfos = 0
    for p in patients:
        for e in p.electrodes:
            for type in e.hfos.keys():
                total_hfos+= len(e.hfos[type])
    print('Total hfos after parsing is {0}'.format(total_hfos))

    #Delete, #Debug
    for p in patients:
        for p2 in patients:
            if p == p2:
                continue
            if p.id == p2.id:
                print('Debug: This should be unreachable')
                assert(False)
    # Models
    models.run_RonO_Model(patients)



if __name__ == "__main__":
    main()
import pymongo

from preprocessing import soz_bool, get_spike_kind, parse_hfos, encode_type_name
import models
from config import HFO_TYPES, DEBUG
from utils import inconsistencies, log, unique_patients
from classes import Database
from sklearn.preprocessing import StandardScaler


def load_patients(hfo_collection, electrodes_collection, hfo_type_names):

    # Todo 'outcome', 'resected',
    common_attr = ['patient_id', 'age', 'file_block', 'electrode', 'loc5', 'soz', 'soz_sc',
                   'type', 'duration', 'fr_duration', 'r_duration',
                   'freq_av', 'freq_pk', 'power_av', 'power_pk']

    patients_by_hfo_type = {hfo_type_name:dict() for hfo_type_name in HFO_TYPES}
    for hfo_type_name in hfo_type_names:
        patients_by_hfo_type[hfo_type_name] = add_patients_by_hfos(patients_by_hfo_type[hfo_type_name],
                                                                   hfo_collection,
                                                                   common_attr,
                                                                   hfo_type_name)
        patients_by_hfo_type[hfo_type_name] = add_empty_blocks(patients_by_hfo_type[hfo_type_name],
                                                               electrodes_collection)

    return patients_by_hfo_type


def add_patients_by_hfos(patients_of_hfo_type, hfo_collection, common_attr, hfo_type_name):

    if hfo_type_name in ['RonO', 'Fast RonO']:
        specific_attributes = ['slow', 'slow_vs', 'slow_angle',
                               'delta', 'delta_vs', 'delta_angle',
                               'theta', 'theta_vs', 'theta_angle',
                               'spindle', 'spindle_vs', 'spindle_angle']
    else:
        specific_attributes = ['spike', 'spike_vs', 'spike_angle']

    hfo_type = encode_type_name(hfo_type_name)
    hfos_cursor = hfo_collection.find(
        filter={'type': hfo_type, 'loc5': 'Hippocampus', 'intraop':'0'},
        projection=common_attr + specific_attributes,
        #sort= [('patient_id', pymongo.ASCENDING), ('electrode', pymongo.ASCENDING)]
    )
    #Unifying types and parsing inconsistencies
    patients_of_hfo_type = parse_hfos(patients_of_hfo_type, hfos_cursor)

    return patients_of_hfo_type


def add_empty_blocks(patients, electrodes_collection):
    # Nota: esto agrega los bloques sin hfos para los (patient,electrode) que tienen otro bloque de ese electrodo
    # con hfos en hipocampo, no considera los patient electrodes con loc5 en hipocampo que no tengan ningun hfo
    empty_blocks_added = 0
    for p in patients.values():
        for e in p.electrodes:
            hfo_empty_blocks = electrodes_collection.find(
                filter={'patient_id': p.id, "$or": [{'electrode': [e.name]}, {'electrode': e.name}]},
                projection=['soz', 'file_block']
            )
            for electrode_rec in hfo_empty_blocks:
                # Consistency for soz
                soz = soz_bool(electrode_rec['soz'])
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
                    file_block = float(electrode_rec['file_block'])
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
    #There are 4 patients that have electrodes in Hippocampus but register 0 hfos in Hippocampus, we skip them
    #['IO015io', 'IO005io', 'IO002io', '3452']
    #with_hfo_in_hip = unique_patients(hfo_collection, {'type': '1', 'intraop':'0', 'loc5': 'Hippocampus'})
    #with_electrodes_in_hip = unique_patients(electrodes_collection, {'loc5': 'Hippocampus'})
    #patients_with_electrodes_but_no_hfos = list(set(with_electrodes_in_hip) - set(with_hfo_in_hip))
    #print(patients_with_electrodes_but_no_hfos)
    type_names_to_run = ['RonS']
    print('HFO types to run: {0}'.format(type_names_to_run))
    print('Loading data from database...')

    patients_by_hfo_type = load_patients(hfo_collection, electrodes_collection, type_names_to_run)

    if DEBUG:
        print('Inconsistencies found while parsing: {0}'.format(inconsistencies))

    for hfo_type_name in type_names_to_run:
        patients_dic = patients_by_hfo_type[hfo_type_name]
        patients = [p for p in patients_dic.values()]
        models.random_forest(patients, hfo_type_name)

        if DEBUG:
            total_hfos = 0
            for p in patients:
                for e in p.electrodes:
                    for type in e.hfos.keys():
                        total_hfos+= len(e.hfos[type])
            print('Total {0} hfos after parsing is {1}'.format(hfo_type_name, total_hfos))


if __name__ == "__main__":
    main()
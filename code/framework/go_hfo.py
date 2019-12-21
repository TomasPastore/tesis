import copy

import pymongo
from matplotlib import pyplot as plt
import numpy as np

import graphics
from classes import Database
from config import (type_names_to_run, EVENT_TYPES, HFO_SUBTYPES, HFO_TYPES,
                    intraop_patients, non_intraop_patients, electrodes_query_fields, hfo_query_fields)
from db_parsing import parse_patients
from phfos import compare_phfo_models, phfo_filter
from utils import encode_type_name

db = Database()
connection = db.get_connection()
db = connection.deckard_new

electrodes_collection = db.Electrodes
electrodes_collection.create_index( [('patient_id', "hashed" )] )
electrodes_collection.create_index( [('electrode', 1)] )
electrodes_collection.create_index( [('type', "hashed" )] )
electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

hfo_collection = db.HFOs
hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')
hfo_collection.create_index( [('patient_id', 1), ('electrode', 1), ('intraop',1), ('type', 1)] )


# Gathers info about patients rate data.
# Event will be considered for the data if it is of type t for any t in EVENT_TYPES
def rate_data(patients_dic, event_types=EVENT_TYPES, evt_filter=None):
    if evt_filter is None:
        evt_filter = {}

    event_rates = []
    labels = []
    elec_count = 0
    event_count = 0
    elec_with_events = 0
    elec_with_pevents = 0
    elec_capture_scores = []
    elec_proportion_scores = []
    phfo_capture_flag = all([e in HFO_TYPES for e in event_types])
    for p in patients_dic.values():
        elec_count += len(p.electrodes)
        for e in p.electrodes:
            event_rates.append(e.get_events_rate(event_types))  # Measured in events/min
            labels.append(e.soz)
            electrode_count = e.get_events_count(event_types)
            if electrode_count > 0:
                event_count += electrode_count
                elec_with_events += 1
            if e.has_pevent(event_types):
                elec_with_pevents += 1

            if phfo_capture_flag:
                elec_capture_scores.append(e.phfo_capture_score(event_types, hfo_collection, evt_filter, p.id))
            elec_proportion_scores.append(e.pevent_proportion_score(event_types))


    rate_info = {
        'evt_rates': event_rates,
        'soz_labels': labels,
        'elec_count': elec_count,
        'evt_count': event_count,
        'p_elec_with_evts': round(100 * (elec_with_events / elec_count), 2),
        'p_elec_with_pevts': round(100 * (elec_with_pevents / elec_count), 2),
        'capture_score': round(100 * np.mean(elec_capture_scores), 2) if len(elec_capture_scores) > 0 else 1, #cuantos phfo caputra en promedio cada electrodo
        'proportion_score': round(100 * np.mean(elec_proportion_scores), 2) #cuantos de los capturados son phfo en promedio entre electrodos
    }
    return rate_info


def all_subtype_names(hfo_type_name):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        subtypes = ['slow', 'delta', 'theta', 'spindle']
    else:
        subtypes = ['spike']
    return subtypes


def all_loc_names(granularity):
    if granularity == 1:
        return ['Right Cerebrum', 'Left Cerebrum']
    elif granularity == 2:
        return ['Limbic Lobe', 'Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe', 'Occipital Lobe']  #
    elif granularity == 3:
        return ['Middle Temporal Gyrus']  # Todo get names
    elif granularity == 4:
        return ['Gray Matter', 'White Matter']
    elif granularity == 5:
        return ['Hippocampus', 'Brodmann area 27', 'Amygdala', 'Brodmann area 21', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37']


def parse_locations(loc_granularity, locations):
    if loc_granularity == 0:
        loc = None
        locations = ['Whole brain']
    else:
        loc = 'loc{i}'.format(i=loc_granularity)
        locations = all_loc_names(granularity=loc_granularity) if locations == 'all' else locations
    return loc, locations


def query_filters(intraop, event_type_names, loc, loc_name, hfo_subtypes=None):
    patient_id_intraop_cond = {'$nin': non_intraop_patients if intraop else intraop_patients}
    encoded_intraop = str(int(intraop))

    elec_filter = {'patient_id': patient_id_intraop_cond}  # 'x' : {'$ne':'-1'}, 'y' : {'$ne':'-1'}, 'z' : {'$ne':'-1'}
    evt_filter = {'patient_id': patient_id_intraop_cond, 'intraop': encoded_intraop,
                    'type': {'$in': [encode_type_name(e) for e in
                                     event_type_names]}}  # 'x' : {'$ne':'-1'}, 'y' : {'$ne':'-1'}, 'z' : {'$ne':'-1'}  '$or': [{'type':'1'}, {'type':'2'}, {'type':'4'}, {'type':'5'} ]

    if hfo_subtypes is not None:
        evt_filter['$or'] = [{'$or': [{subtype: 1} for subtype in hfo_subtypes]},
                               {'type': {'$in': ['Spikes', 'Sharp Spikes']}}]

    if loc is not None:
        elec_filter[loc] = loc_name
        evt_filter[loc] = loc_name

    return elec_filter, evt_filter


def get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr):
    for i in range(len(fpr)):
        if fpr[i] == tolerated_fpr:
            return thresholds[i]
        elif fpr[i] < tolerated_fpr:
            continue
        elif fpr[i] > tolerated_fpr:
            if abs(fpr[i] - tolerated_fpr) <= abs(fpr[i - 1] - tolerated_fpr):
                return thresholds[i]
            else:
                return thresholds[i - 1]


def compare_event_type_rates_by_loc(event_type_names=EVENT_TYPES, loc_granularity=0, locations='all',
                                    intraop=False, filter_phfos=False):
    loc, locations = parse_locations(loc_granularity, locations)
    event_type_data_by_loc = dict()
    for loc_name in locations:
        event_type_data_by_loc[loc_name] = dict()
        elec_filter, evt_filter = query_filters(intraop, event_type_names, loc, loc_name)
        elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor)
        for evt_type_name in event_type_names:
            event_type_data_by_loc[loc_name][evt_type_name] = rate_data(patients_dic, [evt_type_name],
                                                                        evt_filter=evt_filter)

            if filter_phfos:
                patients_dic = phfo_filter(evt_type_name, loc_name, all_patients_dic=patients_dic)
                event_type_data_by_loc[loc_name]['Filtered_' + evt_type_name] = rate_data(patients_dic, [evt_type_name],
                                                                                          evt_filter=evt_filter)

    graphics.event_rate_by_loc(event_type_data_by_loc) #Todo legends
    plt.show()


def compare_subtypes_rate_by_loc(hfo_type_name, subtypes='all', loc_granularity=0, locations='all',
                                 intraop=False, filter_phfos=False):
    subtypes = all_subtype_names(hfo_type_name) if subtypes == 'all' else subtypes
    loc, locations = parse_locations(loc_granularity, locations)
    subtype_data_by_loc = dict()
    for loc_name in locations:
        subtype_data_by_loc[loc_name] = dict()
        for subtype_name in subtypes:

            elec_filter, hfo_filter = query_filters(intraop, [hfo_type_name], loc, loc_name, [subtype_name])
            elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
            hfo_cursor = hfo_collection.find(hfo_filter, projection=hfo_query_fields)
            patients_dic = parse_patients(elec_cursor, hfo_cursor)
            subtype_data_by_loc[loc_name][subtype_name] = rate_data(patients_dic, [hfo_type_name], hfo_filter)

            if filter_phfos:
                patients_dic = phfo_filter(hfo_type_name, loc_name, all_patients_dic=patients_dic)
                subtype_data_by_loc[loc_name]['Filtered_' + hfo_type_name] = rate_data(patients_dic, [hfo_type_name], hfo_filter)

    graphics.event_rate_by_loc(subtype_data_by_loc, zoomed_type=hfo_type_name)#Todo legends
    plt.show()

def build_rate_table(event_type_names=EVENT_TYPES, loc_granularity=0, locations='all', intraop=False):
    loc, locations = parse_locations(loc_granularity, locations)
    event_type_data_by_loc = dict()
    for loc_name in locations:
        print('Location: {0}'.format(loc_name))
        event_type_data_by_loc[loc_name] = dict()
        elec_filter, evt_filter = query_filters(intraop, event_type_names, loc, loc_name)
        elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor)
        for evt_type_name in event_type_names:
            print('Event type: {0}'.format(evt_type_name))
            event_type_data_by_loc[loc_name][evt_type_name] = rate_data(patients_dic, [evt_type_name],
                                                                        evt_filter=evt_filter)
    return event_type_data_by_loc

#EXPERIMENTS
#1 HFO rate of the 4 HFO types colapsed agains spike rate in all brain
def All_brain_HFOs_vs_Spikes(intraop=False):
    import time

    print('Building structures from db...')
    start_time = time.time()
    loc = None
    loc_name = 'Whole brain electrodes'
    event_type_data_by_loc = dict()
    event_type_data_by_loc[loc_name] = dict()
    elec_filter, evt_filter = query_filters(intraop, HFO_TYPES + ['Spikes'], loc, loc_name)
    elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
    hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor)
    print('{0} seconds for parsing patients data'.format(time.time() - start_time))

    print('Calculating Spikes rate data')
    start_time = time.time()
    event_type_data_by_loc[loc_name]['Spikes'] = rate_data(patients_dic, ['Spikes'], evt_filter=evt_filter)
    print('{0} seconds for rate hfo data'.format(time.time() - start_time))

    print('Calculating HFOs rate data')
    start_time = time.time()
    event_type_data_by_loc[loc_name]['HFOs'] = rate_data(patients_dic, HFO_TYPES, evt_filter=evt_filter)
    print('{0} seconds for rate hfo data'.format(time.time() - start_time))

    print('Plotting...')
    start_time = time.time()
    graphics.event_rate_by_loc(event_type_data_by_loc, legend_metrics=['ec', 'pewp', 'ps'])
    print('{0} seconds for the graphic'.format(time.time() - start_time))

    plt.show()

#2) Separating types beats spikes and RonS is best

# 3a) Generate 2 tables for loc, types:
# T1) Proportion of phfos over the total of phfos of all types in the location that are considered by the model
# T2) Percentage of phfos over the total of hfos considered by the model.
# Si tengo buen score en T1 tengo mas cantidad de patologicos contemplados (puede implicar mas fisiologicos tambien),
# Si tengo buen score en T2 tengo mas proporcion de patologicos en mi seleccion
# Si queremos que capture phfos de la mayor cantidad de electrodos posibles (buen T1) y en mayor proporcion patologicos para cada electrodo (buen T2).,
# Podemos hacer una metrica dandole peso a ambas, por ej pscore = 0.4 * s_t1 + 0.6 * s_t2, por ejmplo asumiendo que es mas importante la proporcion final
# Hipotesis, que el hfo rate de mejor en las localizaciones y modelos que tienen asociado un pscore mayor, sugiere que lo que importa es lo patologico.
# 3b) Plotear loc5 distintas incluyendo hippocampo y ver su event rate para los subtipos y spikes, comprobar que dio mejor AUC en los casos que tenia mayor pscore
# tener el pscore para cada loc, modelo, ordenarlos decreciente por pscore y tener dic[loc][model][pscore_place] empezando con 1
#  tabla ordenada por AUC decrecientemente,  Rank loc, modelo, AUC, pscore_place.
#  Si esta primero en rank el pscore place tiene que ser bajo tambien para que 'valga' la hipotesis

def improve_FRonS(loc_granularity=5, locations='all', intraop=False, filter_phfos=False):
    hfo_type_name = 'RonS'
    loc, locations = parse_locations(loc_granularity, locations)
    hfo_type_data_by_loc = dict()
    for loc_name in locations:
        hfo_type_data_by_loc[loc_name] = dict()
        elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc, loc_name)
        elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor)
        hfo_type_data_by_loc[loc_name][hfo_type_name + '_baseline'] = rate_data(patients_dic, [hfo_type_name],
                                                                                evt_filter=evt_filter)
        compare_phfo_models(hfo_type_name, loc_name, copy.deepcopy(patients_dic))
        if filter_phfos:
            patients_dic = phfo_filter(hfo_type_name, loc_name, all_patients_dic=patients_dic)
            hfo_type_data_by_loc[loc_name]['Filtered_' + hfo_type_name] = rate_data(patients_dic, [hfo_type_name],
                                                                                    evt_filter=evt_filter)

    graphics.event_rate_by_loc(hfo_type_data_by_loc)
    plt.show()


# TODO
def improve_FRonO(loc_granularity=2, locations='all', intraop=False, filter_phfos=False):
    hfo_type_name = 'Fast RonO'
    loc, locations = parse_locations(loc_granularity, locations)
    hfo_type_data_by_loc = dict()
    for loc_name in locations:
        hfo_type_data_by_loc[loc_name] = dict()

        elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc, loc_name)

        elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor)
        hfo_type_data_by_loc[loc_name][hfo_type_name + '_baseline'] = rate_data(patients_dic, [hfo_type_name],
                                                                                evt_filter=evt_filter)

        if filter_phfos:
            patients_dic = phfo_filter(hfo_type_name, loc_name, all_patients_dic=patients_dic)
            hfo_type_data_by_loc[loc_name]['Filtered_' + hfo_type_name] = rate_data(patients_dic, [hfo_type_name],
                                                                                    evt_filter=evt_filter)

    graphics.event_rate_by_loc(hfo_type_data_by_loc)
    plt.show()


def main():
    print('HFO types to run: {0}'.format(type_names_to_run))

    #1st experiment
    # Comparar ROCs de HFO rate colapsando subtipos vs spikes
    All_brain_HFOs_vs_Spikes()

    #improve_FRonS(locations=['Hippocampus'])


if __name__ == "__main__":
    main()

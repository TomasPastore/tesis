import copy

import numpy as np
import pymongo
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

import graphics
from classes import Database
from config import (EVENT_TYPES, HFO_TYPES,
                    intraop_patients, non_intraop_patients, electrodes_query_fields, hfo_query_fields)
from db_parsing import parse_patients, parse_locations, encode_type_name
from phfos import phfo_filter
from utils import histograms, phase_coupling_paper_polar, all_subtype_names, all_loc_names

db = Database()
connection = db.get_connection()
db = connection.deckard_new

electrodes_collection = db.Electrodes
electrodes_collection.create_index([('patient_id', "hashed")])
electrodes_collection.create_index([('electrode', 1)])
electrodes_collection.create_index([('type', "hashed")])
electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

hfo_collection = db.HFOs
hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')
hfo_collection.create_index([('patient_id', 1), ('electrode', 1), ('intraop', 1), ('type', 1)])

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


# Gathers info about patients rate data.
# Event will be considered for the data if it is of type t for any t in EVENT_TYPES
def rate_data(patients_dic, event_types=EVENT_TYPES, evt_filter=None, flush=False):
    if evt_filter is None:
        evt_filter = {}
    print('Rating_data {0}'.format(event_types))

    event_rates = []
    labels = []
    elec_count = 0
    event_count = 0
    elec_with_events = 0
    elec_with_pevents = 0
    elec_proportion_scores = []
    for p in patients_dic.values():
        elec_count += len(p.electrodes)
        for e in p.electrodes:
            if flush:
                e.flush_cache(event_types)
            event_rates.append(e.get_events_rate(event_types))  # Measured in events/min
            labels.append(e.soz)
            electrode_count = e.get_events_count(event_types)
            if electrode_count > 0:
                event_count += electrode_count
                elec_with_events += 1
            if e.has_pevent(event_types):
                elec_with_pevents += 1
            prop_score, empty = e.pevent_proportion_score(event_types)
            elec_proportion_scores.append(prop_score)

    if elec_count == 0:
        print(evt_filter)

    pewp = elec_with_pevents / elec_count
    ps = np.mean(elec_proportion_scores)
    rate_info = {
        'evt_rates': event_rates,
        'soz_labels': labels,
        'elec_count': elec_count,
        'evt_count': event_count,
        'p_elec_with_evts': round(100 * (elec_with_events / elec_count), 2),
        'p_elec_with_pevts': round(100 * pewp, 2),
        'proportion_score': ps,  # cuantos de los capturados son phfo en promedio entre electrodos
        'metric_score': pscore(prop_score=ps, pewp=pewp),
        'AUC_ROC': roc_auc_score(labels, event_rates)
    }

    #for k,v in rate_info.items():
    #    print('{0}: {1}'.format(k,v))
    return rate_info


def compare_event_type_rates_by_loc(event_type_names=EVENT_TYPES, loc_granularity=0, locations='all',
                                    intraop=False, filter_phfos=False, tables=None, filter_info=None):
    if filter_phfos:
        assert( isinstance(filter_info, dict) )
        assert('perfect' in filter_info.keys())
        assert('include' in filter_info.keys())

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

            if tables is not None:
                tables['proportion'][loc_name][evt_type_name] = event_type_data_by_loc[loc_name][evt_type_name][
                    'proportion_score']
                tables['metric_score'][loc_name][evt_type_name] = event_type_data_by_loc[loc_name][evt_type_name][
                    'metric_score']
                tables['AUC_ROC'][loc_name][evt_type_name] = event_type_data_by_loc[loc_name][evt_type_name][
                    'AUC_ROC']

            if filter_phfos and evt_type_name != 'Spikes':
                patients_dic = phfo_filter(evt_type_name, patients_dic, target=['model_pat', 'validation_pat'],
                                           tolerated_fpr=None, perfect=True)
                for p in patients_dic.values():
                    for e in p.electrodes:
                        for evt in e.events[evt_type_name]:
                            assert(evt.info['soz'])

                event_type_data_by_loc[loc_name]['Filtered ' + evt_type_name] = rate_data(patients_dic, [evt_type_name],
                                                                                          evt_filter=evt_filter, flush=True)

    #graphics.event_rate_by_loc(event_type_data_by_loc)  # Todo legends
    plt.show()

#Ver si es necesario lo del filter_info
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
                patients_dic = phfo_filter(hfo_type_name, patients_dic, tolerated_fpr=None,
                                           perfect=True)
                subtype_data_by_loc[loc_name]['Filtered_' + hfo_type_name] = rate_data(patients_dic, [hfo_type_name],
                                                                                       hfo_filter)

    graphics.event_rate_by_loc(subtype_data_by_loc, zoomed_type=hfo_type_name)  # Todo legends
    plt.show()

# EXPERIMENTS
# 1 HFO rate of the 4 HFO types colapsed agains spike rate in all brain
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
    graphics.event_rate_by_loc(event_type_data_by_loc, metrics=['ec', 'pewp', 'ps', 'auc'])
    print('{0} seconds for the graphic'.format(time.time() - start_time))

    plt.show()


# 2) Separating types beats spikes and RonS is best
def HFO_subclasses_vs_Spikes(intraop=False):
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

    for event_type_name in EVENT_TYPES:
        if event_type_name == 'Sharp Spikes':
            continue

        print('Calculating {0} rate data'.format(event_type_name))
        start_time = time.time()
        event_type_data_by_loc[loc_name][event_type_name] = rate_data(patients_dic, [event_type_name],
                                                                      evt_filter=evt_filter)
        print('{0} seconds for rate hfo data'.format(time.time() - start_time))

    print('Plotting...')
    start_time = time.time()
    graphics.event_rate_by_loc(event_type_data_by_loc, metrics=['ec', 'pewp', 'ps', 'auc'])
    print('{0} seconds for the graphic'.format(time.time() - start_time))

    plt.show()

# 3 Generate tables for loc, types: proportion, pscore
# T1) Percentage of phfos over the total of hfos considered by the model.
# Si tengo buen score en T1 tengo mas proporcion de patologicos en los canales que tienen al menos un patologico de mi seleccion
# Hipotesis, que el hfo rate de mejor en las localizaciones y modelos que tienen asociado un pscore mayor, sugiere que lo que importa es lo patologico.
def pscore(prop_score, pewp):
    return (pewp + prop_score)/2

def analize_pscores():
    locations = ['Whole brain'] + all_loc_names(2) + all_loc_names(3) + all_loc_names(5)
    tables = {'proportion': {loc: {t: 0 for t in HFO_TYPES} for loc in locations},
              'metric_score': {loc: {t: 0 for t in HFO_TYPES} for loc in locations},
              'AUC_ROC': {loc: {t: 0 for t in HFO_TYPES} for loc in locations}}
    compare_event_type_rates_by_loc(HFO_TYPES, loc_granularity=0, intraop=False, tables=tables)
    compare_event_type_rates_by_loc(HFO_TYPES, loc_granularity=2, intraop=False, tables=tables)
    compare_event_type_rates_by_loc(HFO_TYPES, loc_granularity=3, intraop=False, tables=tables)
    compare_event_type_rates_by_loc(HFO_TYPES, loc_granularity=5, intraop=False, tables=tables)
    graphics.plot_score_table(tables['proportion'], 'proportion_global_table')
    graphics.plot_score_table(tables['metric_score'], 'pscore_global_table')
    graphics.plot_co_metric_auc(tables['metric_score'], tables['AUC_ROC'])


# Argumentar por que elegimos el hipocampo comentando los ROCs y la tabla del hipocampo comparada a otras loc5
# Como algo que sugiere phfo mejora el AUC, tiene sentido que un filtro de phfos mejore el auc porque mejoraria el proportion score
# Ahora vamos a querer filtrar los phfo con un filtro de ml.
# 4th Hippocampus pefect filter
def hippocampus_perfect_filter():
    filter_info = {'perfect': True,
                   'include': ['model_pat', 'validation_pat']}
    compare_event_type_rates_by_loc(event_type_names=HFO_TYPES, loc_granularity=5, locations=['Hippocampus'],
                                    intraop=False, filter_phfos=True, tables=None, filter_info=filter_info)


# 5 th Hippocampus further analysis
# Ver el porcentage de phfo promedio por electrodo, cuantos phfo promedio hay por electrodo, cuantos fhfo,
# scatter de electrodos x = phfop y=fhfop
def hippocampus_RonS_p_analisis():
    intraop = False
    loc, locations = parse_locations(5, ['Hippocampus'])
    elec_filter, evt_filter = query_filters(intraop, ['RonS'], loc, 'Hippocampus')
    elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
    hfo_cursor = hfo_collection.find(evt_filter, projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor)
    sozs = []
    red = []
    green = []
    orange = []
    yellow = []
    p_proportions = []
    pevt_counts = []
    empties = []
    all_empties= []
    with_hfos = []
    for p in patients_dic.values():
        for e in p.electrodes:
            soz = e.soz
            sozs.append(soz)
            phfo = e.has_pevent(['RonS'])
            if phfo:
                pevt_counts.append(e.pevt_count['RonS'])

            prop_score, empty = e.pevent_proportion_score(['RonS'])
            all_empties.append(empty)
            if empty:
                empties.append(prop_score)
            else:
                with_hfos.append(prop_score)
            p_proportions.append(prop_score)
            if soz and phfo:
                red.append(prop_score)
            elif not soz and not phfo:
                green.append(prop_score)
            elif soz and not phfo:
                orange.append(prop_score)
            elif not soz and phfo:
                yellow.append(prop_score)

    # Con lo del assert en parsing db verifica que
    #TODOS LOS ELECTRODOS QUE TIENEN UN PHFO SON SOZ...  en hippocampo
    for i in range(len(sozs)):
        print('SOZ vs props')
        print('SOZ: {0}'.format(sozs[i]))
        print('Empty: {0}'.format(all_empties[i]))
        print('Phfo proportion: {0}'.format(p_proportions[i]))


    print('With hfos props')
    print(with_hfos)
    print('Empty electrodes props')
    print(empties)
    elec_count = len(with_hfos) + len(empties)
    print('Total elec count {0}'.format(elec_count))
    print('Empty proportion {0}'.format(len(empties)/elec_count))

    graphics.barchart(len(red), len(green), len(yellow), len(orange))
    #Barras red, green, orange, yellow
    graphics.histogram(p_proportions, title='Hippocampal RonS pathologic proportion per electrode',
                       label='Electrode pathologic proportion', x_label='Pathologic proportion')

    graphics.histogram(pevt_counts, title='Hippocampal RonS pathologic event count per electrode',
                       label='Electrode pathologic count', x_label='Pathologic count', bins=np.arange(0,2800, 50))


# 6th
# ml_training
#Compara modelos con y sin balanceo.
def hippocampal_RonS_model_v0():
    pass #carga base y llama a predictor v0

#Compara la particion
def hippocampal_RonS_model_V1():
    pass
#Fine tunea
def hippocampal_RonS_model_V2():
    pass

# 7th
# Filtrar y ver si dio mejor

# 8TH
# Establecer un theshold para mejorarlo

def main():
    # print('HFO types to run: {0}'.format(type_names_to_run))
    # phase_coupling_paper(hfo_collection)

    # Thesis

    # 1st experiment
    # Comparar ROCs de HFO rate colapsando subtipos vs spikes
    # All_brain_HFOs_vs_Spikes()

    # 2nd experiment
    # Comparar ROCS de HFO rate aprovechando los subtipos vs spikes
    #HFO_subclasses_vs_Spikes()

    # 3rd experiment
    # Pscore analizes by location
    # Esta es para ver otra forma que tener mas phfo correlaciona con mejor auc roc si tomamos hfo rate
    #analize_pscores()
    #ver los fisiologicos, si son todos soz o todos fisiologicos el filtro no tiene sentido
    # 4th experiment
    # Perfect filter in Hippocampus
    #hippocampus_perfect_filter() #Lo del assert false en db parsing implica que esto sirve porque si tiene un phfo es soz, entonces esto hace que haya una separacion directa. Excepto por los naranjas.
    #El clasificador tiene que pegarle al menos a un bajito porcentaje de los soz por canal recall 0.5 por canal(permito FN),
    # pero no tener falsos positivos, es decir que un fisiologico es tomado por patologico. ALTO FN bajo FP
    # aumentar la pnealizacion del error de tener FP , voy  a dudar de la prediccion que dice que es P, quiero estar seguro.
    # La prob de para decir P  tiene que ser de las mas altas de la distr que dicen p y le pegan
    # 5th
    #hippocampus_RonS_p_analisis()
    # 6th
    hippocampal_RonS_model_v0()
    # ml_filter_training

    # 7th
    # Filtered model with hfo rate

    # 8th
    # Filter threshold to beat baseline

    # improve_FRonS(locations=['Hippocampus'])


if __name__ == "__main__":
    main()

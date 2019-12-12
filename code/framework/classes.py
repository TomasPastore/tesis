import copy

from pymongo import MongoClient
import numpy as np

# Classes
from config import EVENT_TYPES, HFO_TYPES, HFO_SUBTYPES, models_to_run, hfo_query_fields
from utils import encode_type_name


class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")


class Patient():
    def __init__(self, id, age):
        self.id = id
        self.age = age
        self.electrodes = []

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

    def print(self):
        print('Printing patient {0}: '.format(self.id))
        print('\tAge: {0}'.format(self.age))
        print('\tElectrodes: ------------------------------')
        for e in self.electrodes:
            e.print()
        print('------------------------------------------')

    #Esta sirve para hacer particiones balanceadas, porque minimizo el balanceo de grupos de pacientes.
    def get_class_balance(self, hfo_type_name):
        negative_class_count = 0
        positive_class_count = 0
        tot_count=0
        for e in self.electrodes:
            for h in e.hfos[hfo_type_name]:
                tot_count +=1
                if h.info['soz']:
                    positive_class_count +=1
                else:
                    negative_class_count +=1
        return negative_class_count, positive_class_count, tot_count

    # Devuelve la proporcion de eventos que captura la seleccion types, subtypes,
    # sobre el total de hfos de todos los tipos.
    def pevent_percentage_abs(self, event_types, subtypes, hfo_collection, tot_event_filter):
        if ['Spikes']== event_types or ['Sharp Spikes'] == event_types:
            return 100
        else:
            assert('Spikes' not in event_types and 'Sharp Spikes' not in event_types)
            tot_event_filter['type'] = {'$in': [encode_type_name(e) for e in HFO_TYPES]}
            tot_event_filter['soz'] = '1'

            if '$or' in tot_event_filter.keys():
                del tot_event_filter['$or']
            tot = hfo_collection.find(filter=tot_event_filter, projection=hfo_query_fields).count()

            considered_count = 0
            for e in self.electrodes:
                for type in event_types:
                    for h in e.events[type]:
                        if subtypes is None or any([h.info[s] for s in subtypes]):
                            if h.info['soz']:
                                considered_count += 1
            return round(considered_count/ tot, 2)

    #Devuelve la proporcion de patologicos de la seleccion
    def pevent_percentage(self, event_types, subtypes=None):
        tot = 0
        pevents = 0
        for e in self.electrodes:
            for type in event_types:
                for h in e.events[type]:
                    if subtypes is None or any([h.info[s] for s in subtypes]):
                        tot += 1
                        if h.info['soz']:
                            pevents += 1

        return round(pevents/tot, 2)

class Electrode():

    def __init__(self, name, soz, blocks, soz_sc=None, events=None, loc1='empty', loc2='empty', loc3='empty', loc4='empty', loc5='empty'):
        if events is None:
            events = {type: [] for type in EVENT_TYPES}
        self.name = name
        self.soz = soz
        self.blocks = blocks
        self.soz_sc = soz_sc
        self.events = events
        self.loc1 = loc1
        self.loc2 = loc2
        self.loc3 = loc3
        self.loc4 = loc4
        self.loc5 = loc5

    def add(self, event):
        self.events[event.info['type']].append(event)

    #TODO agregar a notas de tesis que es importante detallar como calculamos el hfo rate
    # Gives you the event rate per minute considering events iff it is of any type of the ones listed
    # in event_types and in case of hfo type, it also has to have any subtype of of the ones listed in hfo_subtypes.
    # It also returns how many events were considered to have an idea of the error of the sample
    # Default is all even types (RonS, Spikes, RonO, etc) and NO restriction about subtypes
    def get_events_rate(self, event_types=EVENT_TYPES, hfo_subtypes=None):
        event_count = 0
        block_rates = {block_id:[0, duration] for block_id, duration in self.blocks.items()}
        for event_type in event_types:
            for e in self.events[event_type]:
                    considered = (event_type in ['Spikes', 'Sharp Spikes']) or hfo_subtypes is None
                    if hfo_subtypes is not None:
                        for s in hfo_subtypes:
                            considered = considered or e.info[s]
                    if considered:
                        event_count += 1
                        block_rates[ e.info['file_block'] ][0] += 1
            # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [(rate[0]/(rate[1]/60)) if rate[1] is not None else 0.0 for rate in block_rates.values()]
        block_rates_arr.sort() #avoids num errors
        return sum(block_rates_arr)/len(self.blocks), event_count

    def has_pevent(self, event_types=HFO_TYPES):
        for event_type in event_types:
            for evt in self.events[event_type]:
                    if evt.info['soz']:
                       return True
        return False

    def get_phfo_rate(self, event_types=HFO_TYPES, hfo_subtypes=None):
        event_count = 0
        block_rates = {block_id:[0, duration] for block_id, duration in self.blocks.items()}
        for event_type in event_types:
            for e in self.events[event_type]:
                    considered = (event_type in ['Spikes', 'Sharp Spikes']) or hfo_subtypes is None
                    if hfo_subtypes is not None:
                        for s in hfo_subtypes:
                            considered = considered or e.info[s]
                    if considered and e.info['soz']:
                        event_count += 1
                        block_rates[ e.info['file_block'] ][0] += 1
            # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [(rate[0]/(rate[1]/60)) if rate[1] is not None else 0.0 for rate in block_rates.values()]
        block_rates_arr.sort() #avoids num errors
        return sum(block_rates_arr)/len(self.blocks), event_count


    def print(self):
        print('\t\tPrinting electrode {0}'.format(self.name))
        print('\t\t\tFile_blocks: {0}'.format(self.blocks))
        print('\t\t\tSOZ: {0}'.format(self.soz))
        print('\t\t\tLoc5: {0}'.format(self.loc5))
        print('\t\t\tEvents: {0}'.format([{type_name: len(l) for type_name, l in self.events.items()}]))


class Event():
    def __init__(self, info):
        self.info = info

    def reset_preds(self):
        self.info['prediction'] = {m:[0, 0] for m in models_to_run}
        self.info['proba'] = {m:0 for m in models_to_run}


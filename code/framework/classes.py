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
            for h in e.events[hfo_type_name]:
                tot_count +=1
                if h.info['soz']:
                    positive_class_count +=1
                else:
                    negative_class_count +=1
        return negative_class_count, positive_class_count, tot_count



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
        self.evt_count = {type: {} for type in EVENT_TYPES}
        self.pevt_count = {type: 0 for type in EVENT_TYPES}

    def add(self, event):
        self.events[event.info['type']].append(event)

    def flush_cache(self, event_types):
        for event_type in event_types:
            for block in self.evt_count[event_type].keys():
                self.evt_count[event_type][block] = 0
            self.pevt_count[event_type] = 0

            for evt in self.events[event_type]:
                self.evt_count[event_type][evt.info['file_block']] +=1
                if evt.info['soz']:
                    self.pevt_count[event_type] += 1

    #TODO agregar a notas de tesis que es importante detallar como calculamos el hfo rate
    # Gives you the event rate per minute considering events iff it is of any type of the ones listed
    # in event_types
    def get_events_rate(self, event_types=EVENT_TYPES):
        block_rates = {block_id:[0, duration] for block_id, duration in self.blocks.items()}
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                block_rates[block][0] += count

        # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [(rate[0]/(rate[1]/60)) if rate[1] is not None else 0.0 for rate in block_rates.values()]
        block_rates_arr.sort() #avoids num errors
        return sum(block_rates_arr)/len(self.blocks)

    def get_events_count(self, event_types=EVENT_TYPES):
        result = 0
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                result += count
        return result

    def has_pevent(self, event_types=HFO_TYPES):
        for event_type in event_types:
            if self.pevt_count[event_type] > 0:
                return True
        return False

    # Devuelve la proporcion de eventos patologicos que captura la seleccion de event_types sobre el total de phfos de todos los tipos.
    # Si no hay ningun phfo levanta una excepcion para no considerar al electrodo ya que no habia nada que capturar
    # Recall de phfos
    def phfo_capture_score(self, event_types, hfo_collection, tot_event_filter, pat_id):
        assert (all([e in HFO_TYPES for e in event_types]))
        tot_event_filter['patient_id'] = pat_id
        tot_event_filter['electrode'] = self.name
        tot_event_filter['soz'] = '1'
        tot_event_filter['type'] = {'$in': [encode_type_name(e) for e in HFO_TYPES]}

        if '$or' in tot_event_filter.keys():
            del tot_event_filter['$or']

        tot = hfo_collection.find(filter=tot_event_filter, projection=[]).count()
        captured_count = sum([self.pevt_count[e_type] for e_type in event_types])
        #captured_scores_arr = [self.pevt_count[e_type]/tot if tot > 0 else 1 for e_type in HFO_TYPES]
        #print('Capture score debug')
        #print(captured_scores_arr)
        #s = sum(captured_scores_arr)
        #print(s)
        if tot > 0:
            score = captured_count / tot
            print(
                'Patient {0} Electrode {1} capture score --> {2} out of {3} phfos of all HFO categories. Score: {4}'.format(
                    pat_id,
                    self.name,
                    captured_count,
                    tot,
                    score,
            ))
            return score
        else:
            raise RuntimeWarning('There are no phfos in this channel. Do not consider capture score.')

    # Devuelve la proporcion de patologicos capturados en el electrodo
    # La hipotesis es que entre mayor sea para los que tienen algun phfo, mejor andara el hfo rate
    # Los que no tienen ningun phfo no son considerados porque tendrian hfo rate 0 con filtro perfecto y eso clasificaria bien igual
    def pevent_proportion_score(self, event_types): #Precision de pevents
        tot = sum([sum([v for v in self.evt_count[e_type].values()]) for e_type in event_types])
        pevents = sum([self.pevt_count[e_type] for e_type in event_types])
        prop = pevents / tot if tot>0 else 1
        empty = tot==0
        print('Canal sin ningun hfo')
        # print('In electrode {0} {1} out of {2} captured events are pathologic. E-soz: {3} . Score: {4}'.format(self.name, pevents, tot, self.soz, prop))
        return prop, empty

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


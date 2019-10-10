
from pymongo import MongoClient

# Classes
from config import HFO_TYPES, models_to_run

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")


class Patient():
    def __init__(self, id, age, file_blocks):
        self.id = id
        self.age = age
        self.file_blocks = file_blocks #{block_id:duration or None}
        self.electrodes = []

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

    def print(self):
        print('Printing patient {0}: '.format(self.id))
        print('\tAge: {0}'.format(self.age))
        print('\tFile_blocks: {0}'.format(self.file_blocks))
        print('\tElectrodes: ------------------------------')
        for e in self.electrodes:
            e.print()
        print('------------------------------------------')


class Electrode():

    def __init__(self, name, soz, soz_sc=None, hfos=None, loc1='empty', loc2='empty', loc3='empty', loc4='empty', loc5='empty'):
        if hfos is None:
            hfos = {type: [] for type in HFO_TYPES}
        self.name = name
        self.soz = soz
        self.soz_sc = soz_sc
        self.hfos = hfos
        self.loc1 = loc1
        self.loc2 = loc2
        self.loc3 = loc3
        self.loc4 = loc4
        self.loc5 = loc5

    def add(self, hfo):
        self.hfos[hfo.info['type']].append(hfo)

    def get_hfo_rate(self, hfo_type_name, blocks, subtype=None):
        hfo_count = 0
        block_rates = {block_id:[0, duration] for block_id, duration in blocks.items()}
        for h in self.hfos[hfo_type_name]:
            if subtype is None or h.info[subtype]:
                hfo_count += 1
                block_rates[ h.info['file_block'] ][0] += 1

        # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [(rate[0]/(rate[1]/60)) if rate[1] is not None else 0.0 for rate in block_rates.values()]
        block_rates_arr.sort() #avoids num errors
        return sum(block_rates_arr)/len(blocks), hfo_count

    def print(self):
        print('\t\tPrinting electrode {0}'.format(self.name))
        print('\t\t\tSOZ: {0}'.format(self.soz))
        print('\t\t\tLoc5: {0}'.format(self.loc5))
        print('\t\t\tHFOS: {0}'.format([{type_name: len(l) for type_name, l in self.hfos.items()}]))


class HFO():
    def __init__(self, info):
        self.info = info

    def reset_preds(self):
        self.info['prediction'] = {m:[0, 0] for m in models_to_run}
        self.info['proba'] = {m:0 for m in models_to_run}


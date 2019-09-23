
from pymongo import MongoClient
from config import models_to_run

# Classes
from config import HFO_TYPES

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")


class Patient():
    def __init__(self, id, age=0, file_blocks={1:600}, electrodes=None):
        self.id = id
        self.age = age
        self.file_blocks = file_blocks #{block_id:duration or None}
        if electrodes is None:
            electrodes = []
        self.electrodes = electrodes

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

    def __init__(self, name, soz, soz_sc, hfos=None, loc5=None):
        if hfos is None:
            hfos = {type: [] for type in HFO_TYPES}
        self.name = name
        self.soz = soz
        self.soz_sc = soz_sc
        self.hfos = hfos
        self.loc5 = loc5

    def add(self, hfo):
        self.hfos[hfo.info['type']].append(hfo)

    def get_hfo_rate(self, hfo_type_name, blocks):

        block_rates = {block_id:[0, duration] for block_id, duration in blocks.items()}
        for h in self.hfos[hfo_type_name]:
            block_rates[ h.info['file_block'] ][0] += 1

        block_rate_min_sum = 0
        for block_id, rate in block_rates.items():
            # Note: rate[1] is duration, may be None if no hfo was registered for that block
            block_rate_min_sum += (rate[0]/(rate[1]/60)) if rate[1] is not None else 0

        return block_rate_min_sum/len(blocks)

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


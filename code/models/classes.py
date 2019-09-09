
from pymongo import MongoClient

# Classes
from config import HFO_TYPES

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")


class Patient():
    def __init__(self, id, age, file_blocks, electrodes=None):
        self.id = id
        self.age = age
        self.file_blocks = file_blocks
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

    def print(self):
        print('\t\tPrinting electrode {0}'.format(self.name))
        print('\t\t\tSOZ: {0}'.format(self.soz))
        print('\t\t\tLoc5: {0}'.format(self.loc5))
        print('\t\t\tHFOS: {0}'.format([{type_name: len(l) for type_name, l in self.hfos.items()}]))


class HFO():
    def __init__(self, info):
        self.info = info


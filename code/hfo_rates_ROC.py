import math as mt
import numpy as np
from scipy.stats import circmean

from pymongo import MongoClient
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
#from ggplot import *

class Connect(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")

class ElectrodeInfo():
    def __init__(self, count, total_time, soz):
        self.count = count
        self.total_time = total_time
        self.soz = soz

    def get_events_by_minute(self):
        return (0 if self.total_time is None else self.count/(self.total_time / 60) ) 
    
def soz_bool(db_representation_str):
    return ( True if db_representation_str == "1" else False) 

'''
def print_debug_info(electrodes_info):
    for patient_id, electrodes in electrodes_info.items():
        print('******************* {0} ********************'.format(patient_id))
        maximo = 0
        minimo = 100
        patient_name = patient_id
        soz_array_i = []
        hfo_rates_i = []
       
        for e_name, electrode_info in electrodes.items():
            if electrode_info['hfo_rate'] > maximo:
                maximo = electrode_info['hfo_rate']
            if electrode_info['hfo_rate'] < minimo:
                minimo = electrode_info['hfo_rate']
            print( '\t {0}'.format(e_name))
            print( '\t\t HFO_rate: {0} events/min'.format(round(electrode_info['hfo_rate'],3)))
            print( '\t\t soz: {0}'.format(electrode_info['soz']) )
        print('Patient min hfo rate: {0}'.format(round(minimo,3)))
        print('Patient max hfo rate: {0}'.format(round(maximo,3)))

def save_json(electrodes_info_p):
    import json
    with open("patient_electrode_info.json", "w") as file:
        json.dump(electrodes_info_p, file, indent=4, sort_keys=True)
'''

def get_electrodes_data(electrodes_collection, hfo_collection, hfo_type):

    electrodes =  electrodes_collection.find(filter = {},
                                             projection = ['patient_id', 'file_block', 'soz', 'electrode'])

    hfos = hfo_collection.find(filter = { "$and": [{'type': hfo_type}, {'loc5': 'Hippocampus'}] },
                               projection = ['patient_id', 'file_block', 'r_duration', 'soz', 'electrode'])

    #Calculation of hfo rates and soz arrays accross file_blocks

    #Initialize structures
    patients = dict()
    for e in electrodes:
        patient_id = e['patient_id']
        file_block = e['file_block']
        electrode_name =  e['electrode'][0]
        soz = soz_bool(e['soz'])
        patients[patient_id] = dict()
        patients[patient_id][electrode_name]= dict()
        patients[patient_id][electrode_name][file_block] = ElectrodeInfo(0, None, soz)

        #Contar electrodos
    electrodes_count = 0 
    for pid, p_electrodes in patients.items():
        electrodes_count += len(p_electrodes)            


    for h in hfos:
        patient_id = h['patient_id']
        electrode_name =  h['electrode'][0]
        file_block = h['file_block']
        block_duration = float(h['r_duration'])
        soz = soz_bool(h['soz'])

        if patient_id not in patients.keys():
            patients[patient_id] = dict()

        if electrode_name not in patients[patient_id].keys():
            patients[patient_id][electrode_name] = dict()

        if file_block not in patients[patient_id][electrode_name].keys():
            patients[patient_id][electrode_name][file_block] = ElectrodeInfo(1, block_duration, soz)

        else: #Case: file block was already defined either by another hfo or by Electrodes db 
            patients[patient_id][electrode_name][file_block].count += 1 

            #asserts that every hfo of the same block has the same r_duration
            block_known_time = patients[patient_id][electrode_name][file_block].total_time
            if block_known_time is not None and block_known_time != block_duration:
                raise RuntimeError('block duration should agree among the hfo of the same block') 
            else:
                patients[patient_id][electrode_name][file_block].total_time = block_duration

            patients[patient_id][electrode_name][file_block].soz = patients[patient_id][electrode_name][file_block].soz or soz

    #group accross blocks, adding is better estimation than averge across blocks
    electrodes_info = dict()  #dictionary patientid> electrode > ElectrodeInfo
    soz_array_all = []
    hfo_rates_all = []
    for patient_id, p_electrodes in patients.items():
        electrodes_info[patient_id] = dict()
        for electrode_name, blocks in p_electrodes.items():
            electrode_hfo_rate_sum = 0
            soz = None
            for file_block, block_electrode_info in blocks.items():
                electrode_hfo_rate_sum += block_electrode_info.get_events_by_minute()
                soz = block_electrode_info.soz if soz is None else (soz or block_electrode_info.soz)
            electrodes_info[patient_id][electrode_name] = dict(hfo_rate = electrode_hfo_rate_sum/len(blocks),
                                                               soz = soz
                                                          ) 
            soz_array_all.append(soz)
            hfo_rates_all.append(electrodes_info[patient_id][electrode_name]['hfo_rate'])
    return soz_array_all, hfo_rates_all    

def plot_rocs(oracles, preds, legends):
    plt.title('Receiver Operating Characteristic by HFO type.\nHippocampus electrodes.')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(oracles)):
        fpr, tpr, threshold = metrics.roc_curve(oracles[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, colors[i], label = legends[i]+' AUC = %0.2f' % roc_auc)
   
    plt.legend(loc = 'lower right')
  
    plt.show()

    # method II: ggplot
    #df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    #ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

  
    
def main():
    connection = Connect.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs
    electrodes_collection = db.Electrodes

    soz_arrays= []
    hfo_rates = []
    legends = []
    hfo_type_names = ['ronO','rons', 'spikes', 'fast ronO', 'fast rons', 'sharp spikes']
    for i in range(len(hfo_type_names)):
        soz_array, hfo_rate = get_electrodes_data(electrodes_collection, hfo_collection, hfo_type=str(i+1))
        soz_arrays.append(soz_array)
        hfo_rates.append(hfo_rate)
        legends.append('{hfo_type_name} type, {n_electrodes} electrodes.'.format(hfo_type_name=hfo_type_names[i], 
                                                                                 n_electrodes=len(soz_array)))

    plot_rocs(soz_arrays, hfo_rates, legends)

if __name__ == "__main__":
    main() 



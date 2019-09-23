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


def get_electrodes_data(electrodes_collection, hfo_collection, hfo_type, target_loc, rate_condition):

    
    electrodes_filter ={'loc5':target_loc} 
    hfo_filter = { 'type': hfo_type, 'intraop':'0', 'loc5':target_loc }
   

    electrodes = electrodes_collection.find(filter = electrodes_filter,
                                            projection = ['patient_id', 'electrode', 'file_block', 'soz'])

    
    if (electrodes.count()==0):
        print('Problem in loc: {0}'.format(target_loc))
    
    hfos = hfo_collection.find(filter = hfo_filter,
                               projection = ['patient_id', 'electrode', 'file_block', 'r_duration', 'soz'])

    
    #Calculation of hfo rates and soz arrays accross file_blocks

    #Initialize structures
    patients = dict()
    for e in electrodes:
        patient_id = e['patient_id']
        electrode_name =  e['electrode'][0] if isinstance(e['electrode'],list) else e['electrode']
        file_block = e['file_block']
        soz = soz_bool(e['soz'])
        
        if patient_id not in patients.keys():
            patients[patient_id] = dict()
        if electrode_name not in patients[patient_id].keys():
            patients[patient_id][electrode_name]= dict()
        if file_block not in patients[patient_id][electrode_name].keys():
            patients[patient_id][electrode_name][file_block] = ElectrodeInfo(0, None, soz)
        else:
            patients[patient_id][electrode_name][file_block] = patients[patient_id][electrode_name][file_block] or soz
    

    for h in hfos:
        patient_id = h['patient_id']
        electrode_name =  h['electrode'][0] if isinstance(h['electrode'],list) else h['electrode']
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

    soz_array_all = []
    hfo_rates_all = []
    electrodes_count = 0 
    rates_not_0 = 0
    for patient_id, p_electrodes in patients.items():
        for electrode_name, blocks in p_electrodes.items():
            electrodes_count +=1 
            electrode_hfo_rate_sum = 0
            soz = None
            for file_block, block_electrode_info in blocks.items():
                electrode_hfo_rate_sum += block_electrode_info.get_events_by_minute()
                soz = block_electrode_info.soz if soz is None else (soz or block_electrode_info.soz)

            hfo_rate = electrode_hfo_rate_sum/len(blocks)
            if hfo_rate > 0: #Electrode has hfos
                rates_not_0 +=1
                soz_array_all.append(soz)
                hfo_rates_all.append(hfo_rate)
            else:
                if rate_condition == '':
                    soz_array_all.append(soz)
                    hfo_rates_all.append(hfo_rate)

    hfo_type_names = ['RonO','RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
    print('Data for {0} HFO type in {1} :'.format(hfo_type_names[int(hfo_type)-1], target_loc))
    print('\tLocation target ---> {0}'.format(target_loc))
    print('\tElectrode count ---> {0}'.format(electrodes_count))
    print('\tElectrodes with at least one hfo ---> {0}'.format(rates_not_0))
    print('\tHFO count ---> {0}'.format(hfos.count()))

    return soz_array_all, hfo_rates_all, electrodes_count, rates_not_0, hfos.count()    

def plot_rocs(oracles, preds, legends, title, axe):
    axe.set_title(title, fontdict={'fontsize':10}, loc='left')
    axe.plot([0, 1], [0, 1],'r--')
    axe.set_xlim([0, 1])
    axe.set_ylim([0, 1])
    axe.set_ylabel('True Positive Rate')
    axe.set_xlabel('False Positive Rate')
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(oracles)):
        fpr, tpr, threshold = metrics.roc_curve(oracles[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        axe.plot(fpr, tpr, colors[i], label = legends[i]+'. AUC = %0.2f' % roc_auc)
   
    axe.legend(loc = 'lower right', prop={'size': 6})
  
    # method II: ggplot
    #df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    #ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

  
    
def main():
    connection = Connect.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs
    electrodes_collection = db.Electrodes

    fig = plt.figure()
    fig.suptitle('HFO Rate (events per minute)', fontsize=16)
  
    #locations = {'Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe', 
    #             'Occipital Lobe', 'Posterior Lobe', 'Anterior Lobe', 
    #             'Sub-lobar', 'Limbic Lobe'}

    locations = {'Hippocampus', 'Amygdala', 'Brodmann area 21', 'Brodmann area 27', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37'}

    l = 0
    plot_axe = dict()
    for loc in locations:
        l+=1
        loc_axe = plt.subplot(int('33'+str(l)))
        target_loc = loc
        #target_loc = 'Hippocampus'`
        #target_loc = 'any location'
        #rate_condition = 'with at least one HFO'
        rate_condition = ''

        soz_arrays= []
        hfo_rates = []
        legends = []
        hfo_type_names = ['RonO','RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
        for i in range(len(hfo_type_names)):
            soz_array, hfo_rate, elec_total, elec_rate_not_0, hfo_count = get_electrodes_data(electrodes_collection, 
                                                                                              hfo_collection, 
                                                                                              hfo_type=str(i+1),
                                                                                              target_loc=target_loc,
                                                                                              rate_condition=rate_condition)
            soz_arrays.append(soz_array)
            hfo_rates.append(hfo_rate)

            p_elec_with_hfo=round(100*(elec_rate_not_0/elec_total), 2)
            elec_considered_count = elec_total if rate_condition == '' else elec_rate_not_0
            print_elec_c = '' if rate_condition == '' else ', {0} electrodes'.format(elec_considered_count)
            percentage_str = '{0}% with HFOs'.format(p_elec_with_hfo) if rate_condition == '' \
                                                                      else '{0}% of total count'.format(p_elec_with_hfo)
            legends.append('{hfo_type_name}{print_elec_c}, {percentage_str}. HFO count: {hfo_count}'.format(
                hfo_type_name=hfo_type_names[i], 
                print_elec_c=print_elec_c,
                percentage_str=percentage_str,
                hfo_count=hfo_count
            ))

        elec_str = 'All electrodes' if rate_condition == '' else 'Electrodes'
        title = ('ROC by HFO type based on electrodes HFO rate (events/minutes)\n'
                 '{elec_str} in {target_loc} {rate_condition}\n'
                 'Electrode total count for target location: {elec_total}')

        title = title.format(elec_str=elec_str,target_loc=target_loc, rate_condition=rate_condition, elec_total=elec_total)

        title= "{0}".format(target_loc)
        plot_rocs(soz_arrays, hfo_rates, legends, title, axe=loc_axe)

    plt.show()

if __name__ == "__main__":
    main() 

#Cuando grafico > 0 hfos ''

#ALL zones, all electrodes
#All zones, >0 hfos

#Hip, all electrodes
#Hip >0 hfos
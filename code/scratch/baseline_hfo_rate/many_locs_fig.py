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


def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name


def plot_rocs(oracles, preds, legends, title, axe, elec_total):
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
    axe.text(0.04, 0.89, 'EC: {0}'.format(elec_total), bbox=dict(facecolor='grey', alpha=0.5), transform=axe.transAxes, fontsize=8)
  
    # method II: ggplot
    #df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    #ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

intraop_patients = ['IO001io', 'IO002io', 'IO005io', 'IO006io', 'IO008io', 'IO009io', 'IO010io', 'IO011io', 'IO012io', 'IO013io', 'IO015io', 'IO017', 'IO017io', 'IO018io', 'IO021io', 'IO022io', 'M0423', 'M0580', 'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']
non_intraop_patients = ['2061', '3162', '3444', '3452', '3656', '3748', '3759', '3799', '3853', '3900', '3910', '3943', '3967', '3997', '4002', '4009', '4013', '4017', '4028', '4036', '4041', '4047', '4048', '4050', '4052', '4060', '4061', '4066', '4073', '4076', '4077', '4084', '4085', '4089', '4093', '4099', '4100', '4104', '4110', '4116', '4122', '4124', '4145', '4150', '4163', '4166', '448', '449', '451', '453', '454', '456', '458', '462', '463', '465', '466', '467', '468', '470', '472', '473', '474', '475', '477', '478', '479', '480', '481', '729', '831', 'IO001', 'IO002', 'IO004', 'IO005', 'IO006', 'IO008', 'IO009', 'IO010', 'IO012', 'IO013', 'IO014', 'IO015', 'IO017', 'IO018', 'IO019', 'IO021', 'IO022', 'IO023', 'IO024', 'IO025', 'IO027']


def get_electrodes_data(electrodes_collection, hfo_collection, hfo_type, target_loc, rate_condition, intraop=False):

    if intraop:
        patient_intraop_cond = { '$nin': non_intraop_patients }
        intraop_str = '1'
    else:
        patient_intraop_cond = { '$nin': intraop_patients }
        intraop_str = '0'
    
    electrodes_filter ={'loc5':target_loc, 'patient_id': patient_intraop_cond} 
    hfo_filter = { 'loc5':target_loc, 'type': hfo_type, 'intraop':intraop_str, 'patient_id':patient_intraop_cond}
   

    electrodes = electrodes_collection.find(filter = electrodes_filter,
                                            projection = ['patient_id', 'electrode', 'file_block', 'soz'])

    
    hfos = hfo_collection.find(filter = hfo_filter,
                               projection = ['patient_id', 'electrode', 'file_block', 'r_duration', 'soz'])

    
    #Calculation of hfo rates and soz arrays accross file_blocks

    #Initialize structures
    patients = dict()
    max_block = dict()

    for e in electrodes:
        patient_id = e['patient_id']
        electrode_name =  parse_elec_name(e)
        file_block = int(e['file_block'])
        soz = soz_bool(e['soz'])
        
        if patient_id not in max_block.keys() or max_block[patient_id] < file_block:
            max_block[patient_id] = file_block

        if patient_id not in patients.keys():
            patients[patient_id] = dict()
        if electrode_name not in patients[patient_id].keys():
            patients[patient_id][electrode_name]= dict()
        if file_block not in patients[patient_id][electrode_name].keys():
            patients[patient_id][electrode_name][file_block] = ElectrodeInfo(0, None, soz)
        else:
            patients[patient_id][electrode_name][file_block].soz = patients[patient_id][electrode_name][file_block].soz or soz
    

    for h in hfos:
        patient_id = h['patient_id']
        electrode_name = parse_elec_name(h)
        file_block = int(h['file_block'])
        block_duration = float(h['r_duration'])
        soz = soz_bool(h['soz'])

        if patient_id not in max_block.keys() or max_block[patient_id] < file_block:
            max_block[patient_id] = file_block

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

    #add empty blocks

    added_blocks = 0
    added_pat_blocks = []
    for patient_id, p_electrodes in patients.items():
        for elec_name, blocks in p_electrodes.items():
            for i in range(1, max_block[patient_id]+1):
                    fb = i
                    if fb not in blocks.keys():
                        blocks[fb] = ElectrodeInfo(0, None, False)
                        block_id = patient_id+'_'+str(fb)
                        if block_id not in added_pat_blocks:
                            added_blocks +=1
                            added_pat_blocks.append(block_id)
    #print(sorted(added_pat_blocks))
    soz_array_all = []
    hfo_rates_all = []
    electrodes_count = 0 
    rates_not_0 = 0
    for patient_id, p_electrodes in patients.items():
        for electrode_name, blocks in p_electrodes.items():
            electrodes_count +=1 
            electrode_hfo_rates = []
            soz = None
            for file_block, block_electrode_info in blocks.items():
                electrode_hfo_rates.append( block_electrode_info.get_events_by_minute())
                soz = block_electrode_info.soz if soz is None else (soz or block_electrode_info.soz)

            electrode_hfo_rates.sort()
            hfo_rate = sum(electrode_hfo_rates)/len(blocks)

            if hfo_rate > 0: #Electrode has hfos
                rates_not_0 +=1
                soz_array_all.append(soz)
                hfo_rates_all.append(hfo_rate)
            else:
                if rate_condition == '':
                    soz_array_all.append(soz)
                    hfo_rates_all.append(hfo_rate)

    hfo_type_names = ['RonO','RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
    print('Data for {0} HFO type in {1}:'.format(hfo_type_names[int(hfo_type)-1], target_loc))
    print('\tLocation target ---> {0}'.format(target_loc))
    print('\tElectrode count ---> {0}'.format(electrodes_count))
    print('\tElectrodes with at least one hfo ---> {0}'.format(rates_not_0))
    print('\tHFO count ---> {0}'.format(hfos.count()))

    return soz_array_all, hfo_rates_all, electrodes_count, rates_not_0, hfos.count()    


  
    
def main():
    connection = Connect.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs
    electrodes_collection = db.Electrodes

    fig = plt.figure()
    fig.suptitle('HFO Rate by location (events per minute). Non-intraop electrodes.', fontsize=16)
                                  
    #locations = ['Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe', 
    #             'Occipital Lobe', 'Posterior Lobe', 'Anterior Lobe', 
    #             'Sub-lobar', 'Limbic Lobe']

    locations = ['Hippocampus', 'Amygdala', 'Brodmann area 21', 'Brodmann area 27', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37']

    l = 0
    plot_axe = dict()
    for loc in locations:
        l+=1
        loc_axe = plt.subplot(int('33'+str(l)))
        target_loc = loc
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
                                                                                              rate_condition=rate_condition,
                                                                                              intraop=False)
            soz_arrays.append(soz_array)
            hfo_rates.append(hfo_rate)

            p_elec_with_hfo=round(100*(elec_rate_not_0/elec_total), 2) if elec_total != 0 else 0
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

        title= "{0}".format(target_loc)
        plot_rocs(soz_arrays, hfo_rates, legends, title, loc_axe, elec_total)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

if __name__ == "__main__":
    main() 

#Cuando grafico > 0 hfos ''

#ALL zones, all electrodes
#All zones, >0 hfos

#Hip, all electrodes
#Hip >0 hfos
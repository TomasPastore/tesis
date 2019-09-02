from pymongo import MongoClient

import math as mt
import numpy as np

from scipy.stats import circmean
from astropy.stats import rayleightest
from astropy import units as u
from decimal import Decimal
from matplotlib import pyplot as plt
# from ggplot import *
import sklearn.metrics as metrics

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")

#Electrodes, HFO, patients, metrics, experiments, graph, models, subir al repo

class Electrode():
    def __init__(self, count, total_time, soz):
        self.count = count
        self.total_time = total_time
        self.soz = soz

    def get_events_by_minute(self):
        return (0 if self.total_time is None else self.count / (self.total_time / 60))

    def soz_bool(db_representation_str):
        return (True if db_representation_str == "1" else False)


#Types
# type 4 == fast ronO
# type 5 == fast Rons



def electrodes_data(electrodes_collection, hfo_collection, hfo_type):
    electrodes = electrodes_collection.find(filter={},
                                            projection=['patient_id', 'file_block', 'soz', 'electrode'])

    hfos = hfo_collection.find(filter={"$and": [{'type': hfo_type}, {'loc5': 'Hippocampus'}]},
                               projection=['patient_id', 'file_block', 'r_duration', 'soz', 'electrode'])

    # Calculation of hfo rates and soz arrays accross file_blocks

    # Initialize structures
    patients = dict()
    for e in electrodes:
        patient_id = e['patient_id']
        file_block = e['file_block']
        electrode_name = e['electrode'][0]
        soz = soz_bool(e['soz'])
        patients[patient_id] = dict()
        patients[patient_id][electrode_name] = dict()
        patients[patient_id][electrode_name][file_block] = Electrode(0, None, soz)

    for h in hfos:
        patient_id = h['patient_id']
        electrode_name = h['electrode'][0]
        file_block = h['file_block']
        block_duration = float(h['r_duration'])
        soz = soz_bool(h['soz'])

        if patient_id not in patients.keys():
            patients[patient_id] = dict()

        if electrode_name not in patients[patient_id].keys():
            patients[patient_id][electrode_name] = dict()

        if file_block not in patients[patient_id][electrode_name].keys():
            patients[patient_id][electrode_name][file_block] = Electrode(1, block_duration, soz)

        else:  # Case: file block was already defined either by another hfo or by Electrodes db
            patients[patient_id][electrode_name][file_block].count += 1

            # asserts that every hfo of the same block has the same r_duration
            block_known_time = patients[patient_id][electrode_name][file_block].total_time
            if block_known_time is not None and block_known_time != block_duration:
                raise RuntimeError('block duration should agree among the hfo of the same block')
            else:
                patients[patient_id][electrode_name][file_block].total_time = block_duration

            patients[patient_id][electrode_name][file_block].soz = patients[patient_id][electrode_name][
                                                                       file_block].soz or soz

    # group accross blocks, adding is better estimation than averge across blocks
    electrodes_info = dict()  # dictionary patientid> electrode > ElectrodeInfo
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
            electrodes_info[patient_id][electrode_name] = dict(hfo_rate=electrode_hfo_rate_sum / len(blocks),
                                                               soz=soz
                                                               )
            soz_array_all.append(soz)
            hfo_rates_all.append(electrodes_info[patient_id][electrode_name]['hfo_rate'])
    return soz_array_all, hfo_rates_all

def angle_clusters(collection, amp_step, crit, angle_name):
    angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
    docs = collection.find(crit)
    hfo_count = docs.count()
    angles = []
    for doc in docs:
        angle = doc[angle_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
        angles.append(angle)
        angle_group_id = mt.floor(angle / amp_step)
        angle_grouped[str(angle_group_id)] += 1  # increment count of group

    for k, v in angle_grouped.items():  # normalizing values
        r_value = round((v / hfo_count) * 100, 2)  # show them as relative percentages
        angle_grouped[k] = r_value

    mean_angle = mt.degrees(circmean(angles))
    pvalue = float(rayleightest(np.array(angles) * u.rad))  # doctest: +FLOAT_CMP

    return angle_grouped, amp_step, mean_angle, pvalue, hfo_count

def unique_patients(collection, crit):
    # Unique patients ids for the filter below
    hfos_in_zone = collection.find(crit)
    docs = set()
    for doc in hfos_in_zone:
        docs.add(doc['patient_id'])
    patient_ids = list(docs)
    patient_ids.sort()
    print("Unique patients count in {0}: {1}".format(zone, len(patient_ids)))
    print(patient_ids)

#Graphics

def rose_plot(collection, angle_step=(np.pi / 9)):
    criterion = {'$and': [{'type': "5"}, {'spike': 1}, {'intraop': '0'}, {'soz': '1'}, {'loc5': 'Amygdala'}]}
    count_by_group, step, hfo_count, mean_angle, pvalue  = angle_clusters(collection = collectio,
                                                                             amp_step = angle_step,
                                                                             crit = criterion)
    angles = []
    values = []
    print('{name}. Count by fase group \n'.format(name=loc_name))
    print(count_by_group)
    for k, v in count_by_group.items():
        angles.append(step * float(k))
        values.append(v)

    polar_bar_plot(angles, values, loc_name=loc_name, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count)

def polar_bar_plot(angles, values, loc_name, mean_angle, pvalue, hfo_count):
    # Data
    theta = angles
    heights = values

    # Get an axes handle/object
    fig = plt.figure()
    ax1 = plt.subplot(111, polar=True, )
    bars = ax1.bar(angles,
                   heights,
                   align='center',
                   # color='xkcd:salmon',
                   color=plt.cm.magma(heights),
                   width=0.1,
                   bottom=0.0,
                   edgecolor='k',
                   alpha=0.5,
                   label='HFO count (%)')

    annot = ax1.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                         arrowprops=dict(arrowstyle="->"))

    annot.set_visible(False)

    ## Main tweaks
    # max_count = max(values)
    max_value = max(values)
    radius_limit = max_value + (10 - max_value % 10)  # finds next 10 multiple

    # Angle ticks
    ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    # Radius limits
    ax1.set_ylim(0, max_value)
    # Radius ticks
    ax1.set_yticks(np.linspace(0, radius_limit, 5))

    # Radius tick position in degrees
    # ax1.set_rlabel_position(135)

    # Additional Tweaks
    plt.grid(True)
    plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.35, 1))
    plt.title("Fast RonS SOZ in {location}".format(location=loc_name, ), fontdict={'fontsize': 16}, pad=10)

    raleigh_txt = ('Rayleigh Test \n \n'
                   'P-value: {pvalue} \n'
                   'Mean: {mean}° \n').format(pvalue="{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))

    fig.text(-0.35, 0, raleigh_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    info_txt = 'Total HFO count: {count}'.format(count=hfo_count)
    fig.text(-0.35, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    # fig.text(.5, .00, 'Rayleigh test p-value {pvalue}'.format(pvalue=pvalue), ha='center')

    def update_annot(bar):
        x = bar.get_x() + bar.get_width() / 2.
        y = bar.get_y() + bar.get_height()
        annot.xy = (x, y)
        text = "{c}".format(c=y)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax1:
            for bar in bars:
                cont, ind = bar.contains(event)
                if cont:
                    update_annot(bar)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

#Results

def plot_rocs(oracles, preds, title, legends):
    plt.title('Receiver Operating Characteristic by HFO type.\nHippocampus electrodes.')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(oracles)):
        fpr, tpr, threshold = metrics.roc_curve(oracles[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, colors[i], label=legends[i] + ' AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.show()

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

def soz_bool(db_representation_str):
    return ( True if db_representation_str == "1" else False)

def save_json(electrodes_info_p):
    import json
    with open("patient_electrode_info.json", "w") as file:
        json.dump(electrodes_info_p, file, indent=4, sort_keys=True)


def main():
    db = Database()
    connection = db.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs
    electrodes_collection = db.Electrodes

    #unique_patients(HFOs, {'$and': [{'intraop': '0'},{'loc5' : 'Brodmann area 21'}]})

    #Graphics
    #rose_plot(collection,angle_step, 'Brodmann area 28')

    #Rocs
    soz_arrays = []
    hfo_rates = []
    legends = []
    hfo_type_names = ['ronO', 'rons', 'spikes', 'fast ronO', 'fast rons', 'sharp spikes']

    for i in range(len(hfo_type_names)):
        soz_array, hfo_rate = electrodes_data(electrodes_collection, hfo_collection, hfo_type=str(i + 1))
        soz_arrays.append(soz_array)
        hfo_rates.append(hfo_rate)
        legends.append('{hfo_type_name} type, {n_electrodes} electrodes.'.format(hfo_type_name=hfo_type_names[i],
                                                                                 n_electrodes=len(soz_array)))
    title = 'Title'
    plot_rocs(soz_arrays, hfo_rates, title, legends)

if __name__ == "__main__":
    main()

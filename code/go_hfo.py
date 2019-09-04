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
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, cross_val_score
import pandas as pd
import sklearn.preprocessing, sklearn.decomposition, \
       sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor

hfo_types = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
type_id = {name: (index + 1) for index, name in enumerate(hfo_types)}
def encode_type_name(name):
    return hfo_types.index(name) + 1
def decode_type_name(type_id):
    return hfo_types[type_id - 1]

inconsistencies = {}
def log(text, msg_type=None, patient=None, electrode=None):
    LOG = False
    if LOG:
        print(text)

    if msg_type is not None:
        assert(patient is not None)
        assert(electrode is not None)

        if not patient in inconsistencies.keys():
            inconsistencies[patient] = dict()

        if not electrode in inconsistencies[patient].keys():
            inconsistencies[patient][electrode] = dict()

        if not msg_type in inconsistencies[patient][electrode].keys():
            inconsistencies[patient][electrode][msg_type] = 0

        inconsistencies[patient][electrode][msg_type] += 1


#Classes

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://admin:admin@localhost:27017")

class Patient():
    def __init__(self, id, age, file_blocks, electrodes=[]):
        self.id = id
        self.age = age
        self.file_blocks = file_blocks
        self.electrodes = electrodes

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

default_hfos_dict = {type:[] for type in hfo_types }
class Electrode():
    def __init__(self, name, soz, soz_sc, hfos = default_hfos_dict, loc5=None):
        self.name = name
        self.soz = soz
        self.soz_sc = soz_sc
        self.hfos = hfos
        self.loc5 = loc5

    def add(self, hfo):
        self.hfos[hfo.type].append(hfo)

class HFO():
    def __init__(self, type, file_block, duration, intraop,
                 fr_duration, r_duration,
                 freq_av, freq_pk,
                 power_av, power_pk,):
        self.type = type
        self.type_id = encode_type_name(type)
        self.file_block = file_block
        self.duration = duration
        self.intraop = intraop
        self.fr_duration = fr_duration
        self.r_duration = r_duration
        self.freq_av = freq_av
        self.freq_pk = freq_pk
        self.power_av = power_av
        self.power_pk = power_pk

#Fast queries
def unique_patients(collection, crit):

    # Unique patients ids for filter crit
    # Usage example
    # unique_crit = {'$and': [{'intraop': '0'}, {'loc5': 'Brodmann area 21'}]}
    # unique_patients(hfo_collection, unique_crit)
    hfos_in_zone = collection.find(crit)
    docs = set()
    for doc in hfos_in_zone:
        docs.add(doc['patient_id'])
    patient_ids = list(docs)
    patient_ids.sort()
    print("Unique patients count: {0}".format(len(patient_ids)))
    print(patient_ids)

#Loading Data
def load_hfos(source, type, zone, projection):
    return source.find(filter = {"$and": [{'type': type},
                                          {'loc5': zone}
                                         ]},
                       projection = projection)

def parse_electrodes(electrodes):
    patients = dict()
    for e in electrodes:
        #Patient level
        if not e['patient_id'] in patients.keys():
            patients[e['patient_id']] = Patient(
                id = e['patient_id'],
                age = None if e['age'] == "empty" else int(e['age']),
                file_blocks ={int(e['file_block'])}
            )
        # Check patient consistency
        age = None if e['age'] == "empty" else int(e['age'])
        if age != patients[e['patient_id']].age:
            print('Warning, age should be consistent'\
                  ' between blocks of the same patient')
            if age is not None:
                patients[e['patient_id']].age = age

        patients[e['patient_id']].file_blocks.add(e['file_block'])

        #Electrode level
        electrode = None
        if not e['electrode'][0] in patients[e['patient_id']].electrode_names():
            electrode = Electrode(
                e['electrode'][0],
                soz_bool(e['soz']),
                soz_bool(e['soz_sc'])
            )
        else:
            electrode = [e2 for e2 in patients[e['patient_id']].electrodes if e2.name == e['electrode'][0]][0]
            if (soz_bool(e['soz']) != electrode.soz) or (soz_bool(e['soz_sc']) != electrode.soz_sc):
                log('Warning, soz disagreement among blocks in ' \
                      'the same (patient_id, electrode), ' \
                      'running OR between values', msg_type= 'SOZ_0',
                    patient=e['patient_id'], electrode = e['electrode'][0])
                electrode.soz = electrode.soz or soz_bool(e['soz'])
                electrode.soz_sc = electrode.soz_sc or soz_bool(e['soz_sc'])

        patients[e['patient_id']].add_electrode(electrode)
    return patients

def parse_hfos(patients, hfo_collection, spike_kind):
    for h in hfo_collection:
        patient = None
        # Patient level
        if h['patient_id'] not in patients.keys():
            patient = Patient(
                id = h['patient_id'],
                age = None if h['age'] == "empty" else int(h['age']),
                file_blocks ={int(h['file_block'])},
            )
            patients[h['patient_id']] = patient
        else:
            #Check consistency of patient attributes
            age = None if h['age'] == "empty" else int(h['age'])
            if age != patients[h['patient_id']].age:
                log('Warning, age should be consistent' \
                      ' between blocks of the same patient')
                if age is not None:
                    patients[h['patient_id']].age = age
            patients[h['patients_id']].file_blocks.add(h['file_block'])
            patient = patients[h['patients_id']]

        # Electrode level
        electrode = None
        if not h['electrode'][0] in patient.electrode_names():
            electrode = Electrode(
                h['electrode'][0],
                soz_bool(h['soz']),
                soz_bool(h['soz_sc']),
                loc5 = h['loc5']
            )

            patients[h['patients_id']].add_electrode(electrode)
        else:
            electrode = [e for e in patients[e['patient_id']].electrodes if e.name == h['electrode'][0]][0]
            if (soz_bool(h['soz']) != electrode.soz) or (soz_bool(h['soz_sc']) != electrode.soz_sc):
                log('Warning, soz disagreement among hfos in '\
                      'the same patient_id, electrode, '\
                      'running OR between values', msg_type= 'SOZ_0',
                    patient = h['patient_id'], electrode = h['electrode'][0])
                electrode.soz = electrode.soz or soz_bool(h['soz'])
                electrode.soz_sc = electrode.soz_sc or soz_bool(h['soz_sc'])

        #HFO_level
        hfo = HFO(type = decode_type_name(h['type']),
                  file_block = int(h['file_block']),
                  duration = float(h['duration']),
                  intraop = int(h['intraop']),
                  fr_duration = float(h['fr_duration']),
                  r_duration = float(h['r_duration']),
                  freq_av = float(h['freq_av']),
                  freq_pk = float(h['freq_pk']),
                  power_av = float(h['power_av']),
                  power_pk = float(h['power_pk'])
                  )

        electrode.add(hfo)
    return patients

#Analisis
def segmentate(patients, train_p=0.6, test_p=0.2, val_p=0.2):
    assert(1 - train_p - test_p -val_p == 0)
    patient_count = len(patients)
    train_size = int(patient_count * train_p)
    test_size = int(patient_count * test_p)
    validation_size = test_size
    train_size += patient_count - (train_size + test_size + validation_size)

    train_set = patients[:train_size]
    test_set = patients[train_size:train_size + test_size]
    validation_set = patients[patient_count - validation_size:patient_count]
    return train_set, test_set, validation_set

def run_RonO_Model(all_patients):
    # Select all that have any elec in 'Hippocampus'
    subjects = []
    for p in all_patients:
        for e in p.electrodes:
            if e.loc5 == 'Hippocampus':
                subjects.append(p)

    feature_list = ['duration', 'freq_pk', 'power_pk',
                    'slow', 'slow_vs', 'slow_angle',
                    'delta', 'delta_vs', 'delta_angle',
                    'theta', 'theta_vs', 'theta_angle',
                    'spindle', 'spindle_vs', 'spindle_angle', 'soz']
    features = []
    for s in subjects:
        for e in s.electrodes:
            for h in e.hfos:
                features.append( (h[col_name] for col_name in feature_list) )
    print(np.array(features).shape)
    features = pd.DataFrame(features, columns=feature_list )
    features.describe()
    print('here')
    labels = np.array(features['soz'])
    features = features.drop('soz', axis=1)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                               random_state=42)
    #train, test, validation = segmentate(subjects, train=0.6, test=0.2, val=0.2)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    hits = 0
    total = len(test_labels)
    for i in range(predictions):
        if predictions[i] == test_labels[i]:
            hits+=1

    print('Hitrate: {0}'.format(hits/total))

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)

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
def rose_plot(collection, angle_step=(np.pi / 9)):
    # Usage example
    # rose_plot(collection,angle_step, 'Brodmann area 28')

    loc_name = 'Amygdala'
    angle_type = 'spike'
    criterion = {'$and': [{'type': "5"}, {angle_type: 1}, {'intraop': '0'}, {'soz': '1'}, {'loc5': loc_name}]}
    count_by_group, step, hfo_count, mean_angle, pvalue  = angle_clusters(collection = collection,
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

#Aux
def soz_bool(db_representation_str):
    return (True if db_representation_str == "1" else False)
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


    unique_patients(hfo_collection, {"$and": [
                                        {'type': "1"},
                                        {'loc5': "Hippocampus"}
                                     ]}
                    )
    print('Loading data...')
    electrodes = electrodes_collection.find(filter={},
                                            projection=['patient_id',
                                                        'electrode',
                                                        'file_block',
                                                        'soz',
                                                        'soz_sc',
                                                        'age'])
    patients_dic = parse_electrodes(electrodes)

    target_zone = 'Hippocampus'
    #Todo 'outcome', 'resected',
    common_attr = ['patient_id', 'age', 'file_block',
                   'electrode', 'soz', 'soz_sc', 'loc5',
                   'type', 'duration', 'intraop',
                   'fr_duration', 'r_duration',
                   'freq_av', 'freq_pk',
                   'power_av', 'power_pk',
                   ]

    #Loading by type, Data cleaning and reshaping
    #We have a dictionary of patients
    #Where each patient has electrodes...

    #Type 1
    RonO_attributes = ['slow', 'slow_vs', 'slow_angle',
                       'delta', 'delta_vs', 'delta_angle',
                       'theta', 'theta_vs', 'theta_angle',
                       'spindle', 'spindle_vs', 'spindle_angle']

    RonO = load_hfos(source = hfo_collection,
                     type = encode_type_name('RonO'),
                     zone = target_zone,
                     projection = common_attr + RonO_attributes)
    spike_kind = False
    patients_dic = parse_hfos(patients_dic, RonO, spike_kind)

    '''
    #Type 2
    RonS_attributes = ['spike', 'spike_vs', 'spike_angle']
    RonS = load_hfos(
        source=hfo_collection,
        type = encode_type_name('RonS'),
        zone = target_zone,
        projection = common_attr + RonS_attributes)
    spike_kind = True
    patients_dic = parse_hfos(patients, RonS, spike_kind)

    #Type 4
    Fast_RonO_attributes = RonO_attributes
    Fast_RonO = load_hfos(
        source = hfo_collection,
        type = encode_type_name('Fast_RonO'),
        zone = target_zone,
        projection = common_attr + Fast_RonO_attributes)
    spike_kind = False
    patients_dic = parse_hfos(patients, Fast_RonO, spike_kind)
    #Type 5
    Fast_RonS_attributes = RonS_attributes
    Fast_RonS = load_hfos(
        source=hfo_collection,
        type = encode_type_name('Fast_RonS'),
        zone = target_zone,
        projection = common_attr + Fast_RonS_attributes)
    spike_kind = True
    patients_dic = parse_hfos(patients, Fast_RonS, spike_kind)

    #Type 3
    Spikes_attributes = []
    #Type 6
    Sharp_Spikes_attributes = []
    '''

    #Todo calcular sin y cos de los angulos

    patients = []
    for id, p in patients_dic.items():
        patients.append(p)

    run_RonO_Model(patients)
    print(inconsistencies)
    #Models

if __name__ == "__main__":
    main()

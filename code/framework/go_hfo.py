import pymongo
import graphics
from classes import Database
from config import type_names_to_run, HFO_TYPES, intraop_patients, non_intraop_patients, electrodes_query_fields, hfo_query_fields
from db_parsing import parse_patients
from phfos import compare_phfo_models, phfo_filter
from utils import encode_type_name

db = Database()
connection = db.get_connection()
db = connection.deckard_new

electrodes_collection = db.Electrodes
electrodes_collection.create_index([('type', pymongo.ASCENDING)], unique=False)
electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

hfo_collection = db.HFOs
hfo_collection.create_index([('type', pymongo.ASCENDING)], unique=False)
hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')


def rate_data(patients_dic, hfo_type_name, subtype=None):
    labels = []
    hfo_rates = []
    elec_count = 0
    elec_with_hfos = 0
    hfo_count = 0

    for p in patients_dic.values():
        for e in p.electrodes:
            elec_count += 1
            labels.append(e.soz)
            hfo_rate, elec_hfo_count = e.get_hfo_rate(hfo_type_name, p.file_blocks, subtype)
            hfo_count += elec_hfo_count
            if hfo_rate > 0:
                elec_with_hfos += 1
            hfo_rates.append(hfo_rate)  # Measured in events/min

    rate_info = {
        'soz_labels': labels,
        'hfo_rates': hfo_rates,
        'elec_count': elec_count,
        'p_elec_with_hfo': round(100 * (elec_with_hfos / elec_count), 2),
        'hfo_count': hfo_count
    }
    return rate_info


def all_subtype_names(hfo_type_name):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        subtypes = ['slow', 'delta', 'theta', 'spindle']
    else:
        subtypes = ['spike']
    return subtypes

def all_loc_names(granularity):
    if granularity == 1:
        return ['Right Cerebrum', 'Left Cerebrum']
    elif granularity == 2:
        return ['Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe',
                'Occipital Lobe', 'Posterior Lobe', 'Anterior Lobe',
                'Sub-lobar', 'Limbic Lobe']
    elif granularity == 3:
        return ['Middle Temporal Gyrus']  # Todo get names
    elif granularity == 4:
        return ['Gray Matter', 'White Matter']
    elif granularity == 5:
        return ['Hippocampus', 'Amygdala', 'Brodmann area 21', 'Brodmann area 27', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37']  # Todo complete list

def parse_locations(loc_granularity, locations):
    if loc_granularity == 0:
        loc = None
        locations = ['Whole brain']
    else:
        loc = 'loc{i}'.format(i=loc_granularity)
        locations = all_loc_names(granularity=loc_granularity) if locations == 'all' else locations
    return loc, locations

def query_filters(patient_id_intraop_cond, encoded_intraop, hfo_type_name, loc, loc_name, subtype_name=None):
    elec_filter = {'patient_id': patient_id_intraop_cond}
    hfo_filter = {'patient_id': patient_id_intraop_cond, 'intraop': encoded_intraop,
                  'type': encode_type_name(hfo_type_name)}

    if subtype_name is not None:
        hfo_filter[subtype_name] = 1

    if loc is not None:
        elec_filter[loc] = loc_name
        hfo_filter[loc] = loc_name

    return elec_filter, hfo_filter


def get_soz_confidence_thresh(fpr, thresholds, tolerated_fpr):
    for i in range(len(fpr)):
        if fpr[i] == tolerated_fpr:
            return thresholds[i]
        elif fpr[i] < tolerated_fpr:
            continue
        elif fpr[i] > tolerated_fpr:
            if abs(fpr[i]-tolerated_fpr) <= abs(fpr[i-1]-tolerated_fpr):
                return thresholds[i]
            else:
                return thresholds[i-1]

def compare_hfo_types_rate_by_loc(hfo_type_names=HFO_TYPES, loc_granularity=0, locations='all', intraop=False,
                                  filter_phfos=False):

    loc, locations = parse_locations(loc_granularity, locations)
    encoded_intraop = str(int(intraop))
    patient_id_intraop_cond = {'$nin': non_intraop_patients if intraop else intraop_patients}

    hfo_type_data_by_loc = dict()
    for loc_name in locations:
        hfo_type_data_by_loc[loc_name] = dict()
        for hfo_type_name in hfo_type_names:

            elec_filter, hfo_filter = query_filters(patient_id_intraop_cond, encoded_intraop, hfo_type_name, loc, loc_name)
            elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
            hfo_cursor = hfo_collection.find(hfo_filter, projection=hfo_query_fields)
            patients_dic = parse_patients(elec_cursor, hfo_cursor)

            if filter_phfos:
                fpr, thresh_by_model = compare_phfo_models(hfo_type_name, loc_name, patients_dic)
                model_name = 'XGBoost'
                tolerated_fpr = 0.3
                soz_confidence_thresh = get_soz_confidence_thresh(fpr, thresh_by_model[model_name], tolerated_fpr)
                print('Soz confidence thresh for {t} in {l} with {fp} fpr tolerance: {thresh}'.format(
                t=hfo_type_name, l=loc_name, fp=tolerated_fpr, thresh=soz_confidence_thresh))
                patients_dic = phfo_filter(hfo_type_name,
                                           model_name=model_name,
                                           soz_confidence_thresh= soz_confidence_thresh, #0.2 --> 0.5818337973219778
                                           all_patients_dic=patients_dic,
                                           new_target=None,
                                           )

            hfo_type_data_by_loc[loc_name][hfo_type_name] = rate_data(patients_dic, hfo_type_name)

    graphics.hfo_rate_by_loc(hfo_type_data_by_loc)

#Ask if we will always compare subtypes of the same type or mixed for ex slow from RonO and Fast RonO
def compare_subtypes_rate_by_loc(hfo_type_name, subtypes='all', loc_granularity=0, locations='all', intraop=False,
                                 filter_phfos=False):

    subtypes = all_subtype_names(hfo_type_name) if subtypes == 'all' else subtypes
    loc, locations = parse_locations(loc_granularity, locations)
    encoded_intraop = str(int(intraop))
    patient_id_intraop_cond = {'$nin': non_intraop_patients if intraop else intraop_patients}

    hfo_type_data_by_loc = dict()
    for loc_name in locations:
        hfo_type_data_by_loc[loc_name] = dict()
        for subtype_name in subtypes:

            elec_filter, hfo_filter = query_filters(patient_id_intraop_cond, encoded_intraop, hfo_type_name, loc, loc_name, subtype_name)
            elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
            hfo_cursor = hfo_collection.find(hfo_filter, projection=hfo_query_fields)
            patients_dic = parse_patients(elec_cursor, hfo_cursor)

            if filter_phfos:
                fpr, thresh_by_model = compare_phfo_models(hfo_type_name, loc_name, patients_dic)
                model_name = 'XGBoost'
                tolerated_fpr = 0.2
                soz_confidence_thresh = get_soz_confidence_thresh(fpr, thresh_by_model[model_name], tolerated_fpr)
                print('Soz confidence thresh for {s} {t} in {l} with {fp} fpr tolerance: {thresh}'.format(
                s=subtype_name, t=hfo_type_name, l=loc_name, fp=tolerated_fpr, thresh=soz_confidence_thresh))
                patients_dic = phfo_filter(hfo_type_name,
                                           model_name=model_name,
                                           soz_confidence_thresh=soz_confidence_thresh,
                                           all_patients_dic=patients_dic,
                                           new_target=None)

            hfo_type_data_by_loc[loc_name][subtype_name] = rate_data(patients_dic, hfo_type_name, subtype_name)

    graphics.hfo_rate_by_loc(hfo_type_data_by_loc, zoomed_type=hfo_type_name)


def main():
    print('HFO types to run: {0}'.format(type_names_to_run))

    compare_hfo_types_rate_by_loc(hfo_type_names=type_names_to_run, loc_granularity=0, filter_phfos=True)
    #compare_subtypes_rate_by_loc(hfo_type_name='Fast RonO', loc_granularity=2)


if __name__ == "__main__":
    main()

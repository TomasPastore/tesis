import math as mt

from patient import Patient
from electrode import Electrode
from event import Event
from utils import log
import conf
import pymongo
from artifacts import artifact_filter

########################## PARSING OF FIELDS OF THE DB #########################

# DB constants for formatting
# todo put constants as attributes inside Database class

# Collection level

# Fields of interest from Electrodes collection
# file block : strings from 1 to 6 meaning what block of 10 minutes relative
# to the start of the exam, not parsing needed
electrodes_query_fields = ['patient_id', 'age',  # Patient level
                           'file_block', 'electrode',
                           'loc1', 'loc2', 'loc3', 'loc4', 'loc5',
                           'soz', 'soz_sc', 'x', 'y', 'z']

# Fields of interest from HFOs collection
hfo_query_fields = ['patient_id', 'age',  # Patient level
                    'file_block', 'electrode',  # Elec level
                    'loc1', 'loc2', 'loc3', 'loc4', 'loc5',  # Elec level
                    'soz', 'soz_sc', 'x', 'y', 'z',  # Elec level #todo ask
                    # meaning of _sc
                    'type', 'duration', 'start_t', 'finish_t',
                    'fr_duration', 'r_duration',
                    # ripple and fripple duration, review meaning, I'm using
                    # it to get block duration
                    'freq_av', 'freq_pk', 'power_av', 'power_pk',
                    # PAC, with physiological oscillatory bands todo ask for _vs
                    'slow', 'slow_vs', 'slow_angle',  # 0.5 - 2 HZ
                    'delta', 'delta_vs', 'delta_angle',  # 2.0 - 4 HZ
                    'theta', 'theta_vs', 'theta_angle',  # 4.0 - 8 HZ
                    # 'alpha' is not present but is 8.0 - 12 HZ
                    'spindle', 'spindle_vs', 'spindle_angle',
                    # 16-24 Hz only during stage 2,3 of sleep
                    'spike', 'spike_vs', 'spike_angle', ]

# Patient level todo add tests all patient names == intraop + sleep

all_patient_names = ['2061', '3162', '3444', '3452', '3656', '3748', '3759',
                     '3799', '3853', '3900', '3910', '3943',
                     '3967', '3997', '4002', '4009', '4013', '4017', '4028',
                     '4036', '4041', '4047', '4048', '4050',
                     '4052', '4060', '4061', '4066', '4073', '4076', '4077',
                     '4084', '4085', '4089', '4093', '4099',
                     '4100', '4104', '4110', '4116', '4122', '4124', '4145',
                     '4150', '4163', '4166', '448', '449',
                     '451', '453', '454', '456', '458', '462', '463', '465',
                     '466', '467', '468', '470', '472', '473',
                     '474', '475', '477', '478', '479', '480', '481', '729',
                     '831', 'IO001', 'IO001io', 'IO002',
                     'IO002io', 'IO004', 'IO005', 'IO005io', 'IO006', 'IO006io',
                     'IO008', 'IO008io', 'IO009', 'IO009io',
                     'IO010', 'IO010io', 'IO011io', 'IO012', 'IO012io', 'IO013',
                     'IO013io', 'IO014', 'IO015', 'IO015io',
                     'IO017', 'IO017io', 'IO018', 'IO018io', 'IO019', 'IO021',
                     'IO021io', 'IO022', 'IO022io', 'IO023',
                     'IO024', 'IO025', 'IO027', 'M0423', 'M0580', 'M0605',
                     'M0761', 'M0831', 'M1056', 'M1072', 'M1264']

intraop_patients = ['IO001io', 'IO002io', 'IO005io', 'IO006io', 'IO008io',
                    'IO009io', 'IO010io', 'IO011io', 'IO012io',
                    'IO013io', 'IO015io', 'IO017io', 'IO018io', 'IO021io',
                    'IO022io', 'M0423', 'M0580',
                    'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']

non_intraop_patients = ['2061', '3162', '3444', '3452', '3656', '3748', '3759',
                        '3799', '3853', '3900', '3910', '3943',
                        '3967', '3997', '4002', '4009', '4013', '4017', '4028',
                        '4036', '4041', '4047', '4048', '4050',
                        '4052', '4060', '4061', '4066', '4073', '4076', '4077',
                        '4084', '4085', '4089', '4093', '4099',
                        '4100', '4104', '4110', '4116', '4122', '4124', '4145',
                        '4150', '4163', '4166', '448', '449',
                        '451', '453', '454', '456', '458', '462', '463', '465',
                        '466', '467', '468', '470', '472',
                        '473', '474', '475', '477', '478', '479', '480', '481',
                        '729', '831', 'IO001', 'IO002', 'IO004',
                        'IO005', 'IO006', 'IO008', 'IO009', 'IO010', 'IO012',
                        'IO013', 'IO014', 'IO015',
                        'IO018', 'IO019', 'IO021', 'IO022', 'IO023', 'IO024',
                        'IO025', 'IO027']

removed_cause_was_in_both = 'IO017'


# todo change default 0 to None and warnings if age is considered as feature
def parse_age(doc):  # AGE 0 represents empty
    return 0.0 if doc['age'] == "empty" else float(doc['age'])


# Electrode level

def parse_elec_name(doc):
    if isinstance(doc['electrode'],
                  list):  # Its always length of 1 the array at list a sample but just in case
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise ValueError('Unknown type for electrode name')
    if e_name is None:  # This shouldn't happen, Compass schema says it doesn't but is a sample, just to check...
        raise ValueError('Channel without name')

    return e_name


def encode_type_name(name):
    # Returns the type code number in the db for the event name from string
    return str(EVENT_TYPES.index(name) + 1)


def decode_type_name(type_id):
    # Returns the event name string from the db code number
    return EVENT_TYPES[int(type_id) - 1]


def parse_soz_field(db_representation_str):  # SOZ is True if soz = "1" in db
    return (True if db_representation_str == "1" else False)


# Coordinates x y z
def parse_coord(param):  # -1 Represents empty, consider filtering
    # Segun la muestra de compass no hay listas en este campo pero creo que habia saltado error asi que pongo el valor de null en la db
    parsed_coord = None if (
            isinstance(param, list) or param == '-1') else float(param)
    # if parsed_coord is None: #Just to check a buggy branch with coord=[] from db
    #    print('Coord parsed to None from elec: {0}'.format(param))
    #    print(type(param))
    return parsed_coord


# LOCATION TAGS
WHOLE_BRAIN_L0C = 'Whole Brain'


def parse_loc(doc, i):
    # TODO test this and change code
    # Posible doc inputs : ['empty'], ['CH NAME'], [],, [[]], [''], 'empty', 'CH name', [[['empty']]]
    # ['empty'], 'empty', [], ['']  --> 'empty'
    # ['CH NAME'] 'CH NAME' --> 'CH NAME'

    if isinstance(doc['loc{0}'.format(i)], str):  # Is string
        if len(doc['loc{0}'.format(i)]) > 0:  # Case = 'empty' or 'CH NAME'
            loc = doc['loc{0}'.format(i)]
        else:  # Case = ''
            loc = 'empty'

    elif isinstance(doc['loc{0}'.format(i)], list):  # Is a List
        if len(doc['loc{0}'.format(
                i)]) > 0:  # Case = ['empty'], ['CH NAME'], [''], [list]
            if isinstance(doc['loc{0}'.format(i)][0],
                          str):  # Case = ['empty'], ['CH NAME'], ['']
                if len(doc['loc{0}'.format(i)][0]) > 0:
                    loc = doc['loc{0}'.format(i)][0]
                else:  # ['']
                    loc = 'empty'
            elif isinstance(doc['loc{0}'.format(i)][0], list):  # Case = [list]
                if len(doc['loc{0}'.format(i)][0]) == 0:  # Case # [[]]
                    loc = 'empty'
                else:
                    raise RuntimeError('Unknown type for loc')
        else:  # []
            loc = 'empty'
    else:
        raise RuntimeError('Unknown type for loc')

    assert (isinstance(loc, str))  # 'empty ' represents location null
    return loc


def all_loc_names(granularity):  # all to test
    if granularity == 0:
        return ['Whole Brain']
    if granularity == 2:
        return ['Frontal Lobe', 'Temporal Lobe', 'Parietal Lobe',
                'Limbic Lobe', 'Occipital Lobe']
    elif granularity == 3:
        # removed: 'Extra-Nuclear',
        return ['Angular Gyrus', 'Anterior Cingulate', 'Caudate',
                'Cerebellar Tonsil', 'Cingulate Gyrus', 'Claustrum', 'Culmen',
                'Cuneus', 'Declive', 'Fusiform Gyrus',
                'Inferior Frontal Gyrus', 'Inferior Occipital Gyrus',
                'Inferior Parietal Lobule', 'Inferior Temporal Gyrus', 'Insula',
                'Lentiform Nucleus', 'Lingual Gyrus', 'Medial Frontal Gyrus',
                'Middle Frontal Gyrus', 'Middle Occipital Gyrus',
                'Middle Temporal Gyrus', 'Orbital Gyrus', 'Paracentral Lobule',
                'Parahippocampal Gyrus', 'Postcentral Gyrus',
                'Posterior Cingulate', 'Precentral Gyrus', 'Precuneus',
                'Pyramis', 'Sub-Gyral', 'Subcallosal Gyrus',
                'Superior Frontal Gyrus', 'Superior Occipital Gyrus',
                'Superior Parietal Lobule', 'Superior Temporal Gyrus',
                'Supramarginal Gyrus', 'Thalamus', 'Transverse Temporal Gyrus',
                'Tuber', 'Uncus']


    elif granularity == 5:
        return ['Amygdala', 'Brodmann area 1', 'Brodmann area 10',
                'Brodmann area 11', 'Brodmann area 13', 'Brodmann area 17',
                'Brodmann area 18', 'Brodmann area 19', 'Brodmann area 2',
                'Brodmann area 20', 'Brodmann area 21', 'Brodmann area 22',
                'Brodmann area 23', 'Brodmann area 24', 'Brodmann area 25',
                'Brodmann area 27', 'Brodmann area 28', 'Brodmann area 29',
                'Brodmann area 3', 'Brodmann area 30', 'Brodmann area 31',
                'Brodmann area 32', 'Brodmann area 33', 'Brodmann area 34',
                'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37',
                'Brodmann area 37 ', 'Brodmann area 38', 'Brodmann area 39',
                'Brodmann area 4', 'Brodmann area 40', 'Brodmann area 41',
                'Brodmann area 42', 'Brodmann area 44', 'Brodmann area 45',
                'Brodmann area 46', 'Brodmann area 47', 'Brodmann area 5',
                'Brodmann area 6', 'Brodmann area 7', 'Brodmann area 8',
                'Brodmann area 9', 'Caudate Body', 'Caudate Head',
                'Caudate Tail', 'Hippocampus', 'Lateral Globus Pallidus',
                'Pulvinar', 'Putamen']
    else:
        return []


def print_all_locations(elec_collection, evt_collection):
    print(elec_collection.distinct('loc2'))
    print(elec_collection.distinct('loc3'))
    print(elec_collection.distinct('loc5'))
    print(evt_collection.distinct('loc2'))
    print(evt_collection.distinct('loc3'))
    print(evt_collection.distinct('loc5'))


# locations in MNI space loc1, loc2, loc3, loc4, loc5
def preference_locs(granularity):
    if granularity == 0:
        return ['Whole Brain']
    elif granularity == 2:
        return ['Frontal Lobe', 'Temporal Lobe', 'Limbic Lobe',
                'Parietal Lobe', 'Occipital Lobe', ]
    elif granularity == 3:
        return ['Parahippocampal Gyrus',
                'Middle Temporal Gyrus', 'Superior Temporal Gyrus',
                'Sub-Gyral', 'Uncus', 'Fusiform Gyrus']  # el sup frontal tiene
        # 4 pat soz nada mas
    elif granularity == 5:
        return ['Hippocampus', 'Amygdala',
                'Brodmann area 20', 'Brodmann area 21',
                'Brodmann area 28', 'Brodmann area 36']  # BA 6 BA35 #21 y 28
        # dan bajo AUC, 20 0.74 con RonS, la 35 vale la pena para RonS FRonS
        # 0.78 y 0.79
    else:
        raise NotImplementedError('Implement granularity locs')


def get_granularity(loc):
    if loc == None or loc == 'Whole Brain':
        return 0
    for i in range(6):
        if loc in all_loc_names(i):
            return i
    raise RuntimeError('Unknown location name: {0}'.format(loc))


# Returns the loc field and list of locations for the parameters
def get_locations(loc_granularity, locations='all'):
    if loc_granularity == 0:
        loc = None
        locations = ['Whole Brain']
    else:
        loc = 'loc{i}'.format(i=loc_granularity)
        locations = preference_locs(
            granularity=loc_granularity) if locations == 'all' else locations
    return loc, locations


# Events level
def parse_freq(info, freq_name, event):
    info[freq_name] = bool(event[freq_name])
    fq_vs_name = freq_name + '_vs'
    fq_vs = event[fq_vs_name]
    info[fq_vs_name] = None if isinstance(fq_vs, list) or fq_vs is None \
        else float(fq_vs)
    fq_angle_name = freq_name + '_angle'
    fq_angle = event[fq_angle_name]
    info[fq_angle_name] = None if isinstance(fq_angle, list) or fq_angle is None \
        else float(fq_angle)

    # type of the event


EVENT_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS',
               'Sharp Spikes']
HFO_TYPES = ['Fast RonO', 'Fast RonS', 'RonO', 'RonS']
PAC = ['delta', 'theta', 'slow', 'spindle', 'spike']


def all_pac_names(hfo_type_name):  # For HFOs to get PAC
    if hfo_type_name in ['RonO', 'Fast RonO']:
        subtypes = ['slow',  # 0.5 - 2 HZ
                    'delta',  # 2.0 - 4 HZ
                    'theta',  # 4.0 - 8 HZ
                    # alpha is not present but is 8.0 - 12 HZ
                    'spindle']  # 16-24 Hz only during stage 2,3 of sleep
    else:
        subtypes = ['spike']  # epileptiform discharge
    return subtypes


################################################################################

class Database(object):
    @staticmethod
    def get_connection():
        return pymongo.MongoClient("mongodb://localhost:27017")

    def get_collections(self):
        connection = self.get_connection()
        db = connection.deckard_new

        electrodes_collection = db.Electrodes
        electrodes_collection.create_index([('patient_id', "hashed")])
        electrodes_collection.create_index([('electrode', 1)])
        electrodes_collection.create_index([('type', "hashed")])

        electrodes_collection.create_index([('loc5', pymongo.TEXT)],
                                           default_language='english')

        event_collection = db.HFOs
        event_collection.create_index([('loc5', pymongo.TEXT)],
                                      default_language='english')
        event_collection.create_index(
            [('patient_id', 1), ('electrode', 1), ('intraop', 1), ('type', 1)])
        return electrodes_collection, event_collection


# Note we load untagged from db, solve inconsistencies in memory and then
# remove electrodes with null coords or empty locs.
def load_patients(electrodes_collection, evt_collection, intraop,
                  loc_granularity, locations,
                  event_type_names=None,
                  models_to_run=conf.ML_MODELS_TO_RUN,
                  subtypes_from_db=None,  # Default None takes any subtype
                  load_untagged_coords_from_db=True,
                  load_untagged_loc_from_db=True,
                  restrict_to_tagged_coords=True,
                  restrict_to_tagged_locs=True,
                  remove_elec_artifacts=True):
    # Creates a patients loc dictionary for each location as given as parameter
    # Makes a db find for each location
    print('Loading patients from db...')

    loc, locations = get_locations(loc_granularity, locations)
    print('Locations: {0}'.format(locations))

    if event_type_names is None:
        event_type_names = EVENT_TYPES
    print('Event type names: {0}'.format(event_type_names))

    patients_by_loc = dict()
    for loc_name in locations:
        elec_filter, evt_filter = query_filters(intraop, event_type_names, loc,
                                                loc_name,
                                                hfo_subtypes=subtypes_from_db,
                                                allow_null_coords=load_untagged_coords_from_db,
                                                allow_empty_loc=load_untagged_loc_from_db)

        elec_cursor = electrodes_collection.find(elec_filter,
                                                 projection=electrodes_query_fields)
        hfo_cursor = evt_collection.find(evt_filter,
                                         projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor,
                                      event_type_names,
                                      models_to_run,
                                      restrict_to_tagged_coords,
                                      restrict_to_tagged_locs)

        if loc_name == 'Whole Brain' and remove_elec_artifacts:

            from ml_hfo_classifier import artifact_filter

            for hfo_type in ['Fast RonO']:
                patients_dic = artifact_filter(hfo_type, patients_dic)

        patients_by_loc[loc_name] = patients_dic

    return patients_by_loc


# Creates the pymongo filters
def query_filters(intraop, event_type_names, loc, loc_name, hfo_subtypes=None,
                  allow_null_coords=True,
                  allow_empty_loc=True):
    patient_id_intraop_cond = {
        '$in': intraop_patients if intraop else non_intraop_patients}
    encoded_intraop = str(int(intraop))  # int(True) == 1

    elec_filter = {
        'patient_id': patient_id_intraop_cond,
    }
    evt_filter = {'patient_id': patient_id_intraop_cond,
                  'intraop': encoded_intraop,
                  # probably this intraop is actually not needed but doesnt harm anyway
                  'type': {'$in': [encode_type_name(e) for e in
                                   event_type_names],
                           },  # order doesnt matter

                  }
    if hfo_subtypes is not None:  # This says that it belongs to a subtype of a type in evt filter or the event is a Spike that doesnt have subtypes

        # O soy un hfo o un spike
        evt_filter['$or'] = [{'$and': [{'$or': [{subtype: 1} for subtype in
                                                hfo_subtypes]},
                                       {'freq_av': {'$ne': 0}},
                                       {'freq_pk': {'$ne': 0}},
                                       {'power_av': {'$ne': 0}},
                                       {'power_pk': {'$ne': 0}},

                                       ]},
                             {'type': {'$in': [encode_type_name(t) for t in
                                               ['Spikes', 'Sharp Spikes']]}}
                             ]
    else:
        evt_filter['$or'] = [{'$and': [{'freq_av': {'$ne': 0}},
                                       {'freq_pk': {'$ne': 0}},
                                       {'power_av': {'$ne': 0}},
                                       {'power_pk': {'$ne': 0}}
                                       ]},
                             {'type': {'$in': [encode_type_name(t) for t in
                                               ['Spikes', 'Sharp Spikes']]}}
                             ]
    if not allow_null_coords:
        elec_filter['x'] = {"$ne": "-1"}
        elec_filter['y'] = {"$ne": "-1"}
        elec_filter['z'] = {"$ne": "-1"}
        evt_filter['x'] = {"$ne": "-1"}
        evt_filter['y'] = {"$ne": "-1"}
        evt_filter['z'] = {"$ne": "-1"}
    if not allow_empty_loc:
        elec_filter['loc2'] = {"$ne": "empty"}
        elec_filter['loc3'] = {"$ne": "empty"}
        elec_filter['loc5'] = {"$ne": "empty"}
        evt_filter['loc2'] = {"$ne": "empty"}
        evt_filter['loc3'] = {"$ne": "empty"}
        evt_filter['loc5'] = {"$ne": "empty"}

    if loc is not None:
        if allow_empty_loc:
            elec_filter[loc] = {'$in': ['empty', loc_name]}
            evt_filter[loc] = {'$in': ['empty', loc_name]}
        else:
            elec_filter[loc] = loc_name
            evt_filter[loc] = loc_name

    print('Printing filters...')
    print(elec_filter)
    print(evt_filter)
    return elec_filter, evt_filter


# Main parsing functions
def parse_patients(electrodes_cursor, hfo_cursor, event_type_names,
                   models_to_run, rm_xyz_null, rm_loc_empty):
    # Returns a dictionary of patients with electrodes and events loaded
    # from the cursors and solving inconsistencies
    print('Parsing db patients...')
    patients_dic = dict()
    parse_electrodes(patients_dic, electrodes_cursor, event_type_names)
    assert (hfo_cursor.count() > 0)
    parse_events(patients_dic, hfo_cursor, event_type_names, models_to_run,
                 remove_xyz_null=rm_xyz_null,
                 remove_loc_empty=rm_loc_empty)
    return patients_dic


def parse_electrodes(patients, elec_cursor, event_type_names):
    # Populates patients dic with electrode entries (From Electrodes collection)
    # Modifies patients
    for e in elec_cursor:
        # Patient level
        patient_id = e['patient_id']
        if patient_id not in patients.keys():
            patient = Patient(id=patient_id, age=parse_age(e))
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(e)
            if age != patient.age:
                log(
                    'Warning, age should agree among electrodes of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

        # Electrode level
        e_name = parse_elec_name(e)
        file_block = int(e['file_block'])
        x, y, z = parse_coord(e['x']), parse_coord(e['y']), parse_coord(e['z'])
        loc1, loc2, loc3, loc4, loc5 = parse_loc(e, 1), parse_loc(e, 2), \
                                       parse_loc(e, 3), parse_loc(e, 4), \
                                       parse_loc(e, 5)

        first_time_seen = e_name not in patient.electrode_names()
        if first_time_seen:
            electrode = Electrode(name=e_name, soz=parse_soz_field(e['soz']),
                                  blocks={file_block: None}, x=x, y=y, z=z,
                                  soz_sc=(e['soz_sc']),
                                  loc1=loc1, loc2=loc2, loc3=loc3, loc4=loc4,
                                  loc5=loc5,
                                  event_type_names=event_type_names)
            patient.add_electrode(electrode)
        else:  # The electrode exists probably because there are many blocks for the electrodes
            electrode = patient.get_electrode(e_name)
            # Check consistency
            assert (electrode.name == e_name and patient.id == e['patient_id'])

            if parse_soz_field(e['soz']) != electrode.soz:
                log(msg=('Warning, soz disagreement among blocks of '
                         'the same patient_id, electrode, '
                         'running OR between values'),
                    msg_type='elec_blocks_SOZ_conflict',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                electrode.soz = electrode.soz or parse_soz_field(e['soz'])
            electrode.soz_sc = electrode.soz_sc or parse_soz_field(e['soz_sc'])

            if file_block not in electrode.blocks.keys():
                electrode.blocks[
                    file_block] = None  # here we will save the time of the block that is given in the HFO db...
            ##Locations
            if electrode.loc1 == 'empty':
                electrode.loc1 = loc1
            if electrode.loc2 == 'empty':
                electrode.loc2 = loc2
            if electrode.loc3 == 'empty':
                electrode.loc3 = loc3
            if electrode.loc4 == 'empty':
                electrode.loc4 = loc4
            if electrode.loc5 == 'empty':
                electrode.loc5 = loc5

            if electrode.x is None:
                electrode.x = x
            if electrode.y is None:
                electrode.y = y
            if electrode.z is None:
                electrode.z = z
            if x is not None and electrode.x != x:
                raise RuntimeError(
                    'X disagreement (both not null) in electrode blocks')

            if loc5 != 'empty' and electrode.loc5 != loc5:
                RuntimeError(
                    'This shouldnt happen, says that blocks of the same electrode have dif names none empty')
                log(msg=(
                    'Warning, loc5 disagreement among blocks for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                )
                if loc5 == 'Hippocampus':  # Priority
                    electrode.loc5 = loc5


def parse_events(patients, event_collection, event_type_names, models_to_run,
                 remove_xyz_null=False,
                 remove_loc_empty=False):
    # Populates patients dic with event entries (From HFO collection)
    # Modifies patients
    for evt in event_collection:
        # Patient level
        patient_id = evt['patient_id']
        if patient_id not in patients.keys():
            # raise RuntimeWarning('The parse_electrodes may should have had the patient created first')
            patient = Patient(id=patient_id, age=parse_age(
                evt))
            # moment
            patients[patient.id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(evt)
            if age != patient.age:
                log(
                    'Warning, age should be consistent among events of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

        assert (patient is patients[
            patient_id])  # They refer to the same object, but we use the short one
        # Electrode level
        e_name = parse_elec_name(evt)
        soz = parse_soz_field(evt['soz'])
        file_block = int(evt['file_block'])
        block_duration = float(evt['r_duration'])
        x, y, z = parse_coord(evt['x']), parse_coord(evt['y']), parse_coord(
            evt['z'])
        loc1, loc2, loc3, loc4, loc5 = parse_loc(evt, 1), parse_loc(evt,
                                                                    2), parse_loc(
            evt, 3), parse_loc(evt,
                               4), parse_loc(
            evt, 5)

        if e_name not in patient.electrode_names():
            # raise RuntimeWarning('The parse_electrodes should have created the electrode for this event')
            # The code below would create a new a electrode from the HFO info
            electrode = Electrode(name=e_name, soz=soz,
                                  blocks={file_block: block_duration},
                                  x=x, y=y, z=z,
                                  soz_sc=parse_soz_field(evt['soz_sc']),
                                  loc1=loc1, loc2=loc2, loc3=loc3, loc4=loc4,
                                  loc5=loc5,
                                  event_type_names=event_type_names)
            patient.add_electrode(electrode)
        else:
            electrode = patient.get_electrode(e_name)
            # Check consistency

            # SOZ
            if soz != electrode.soz:
                log(msg=('Warning, soz disagreement among event and electrode'
                         'running OR between values'),
                    msg_type='elec_evt_SOZ_conflict',
                    patient=patient.id,
                    electrode=electrode.name
                    )
            electrode.soz = electrode.soz or soz  # Fixs db bug of event nsoz and electrode.soz
            electrode.soz_sc = electrode.soz_sc or parse_soz_field(
                evt['soz_sc'])

            # File block and duration
            if file_block not in electrode.blocks.keys() or electrode.blocks[
                file_block] is None:
                electrode.blocks[file_block] = block_duration
            else:
                if (block_duration != electrode.blocks[file_block]):
                    raise NotImplementedError(
                        'Implement which duration must be saved if they differ.')

            # X Y Z
            if electrode.x is None:
                electrode.x = x
            if electrode.y is None:
                electrode.y = y
            if electrode.z is None:
                electrode.z = z
            if x is not None and electrode.x != x:
                raise RuntimeError(
                    'X disagreement (both not null) in event/electrode')

            ##Locations
            if electrode.loc1 == 'empty':
                electrode.loc1 = loc1
            if electrode.loc2 == 'empty':
                electrode.loc2 = loc2
            if electrode.loc3 == 'empty':
                electrode.loc3 = loc3
            if electrode.loc4 == 'empty':
                electrode.loc4 = loc4
            if electrode.loc5 == 'empty':
                electrode.loc5 = loc5

        # Elec count update
        evt_type = decode_type_name(evt['type'])
        if file_block not in electrode.evt_count[evt_type].keys():
            electrode.evt_count[evt_type][file_block] = 1
        else:
            electrode.evt_count[evt_type][file_block] += 1

        # HFO_level
        info = dict(
            x=x,
            y=y,
            z=z,
            prediction={m: [] for m in models_to_run},
            # for saving results of a model
            proba={m: [] for m in models_to_run},
            # for saving results of a model
            soz=soz,
            type=decode_type_name(evt['type']),
            file_block=int(evt['file_block']),
            start_t=float(evt['start_t']),
            finish_t=float(evt['finish_t']),
            duration=float(evt['duration']) * 1000,  # duration to milliseconds
            fr_duration=float(evt['fr_duration']),
            r_duration=float(evt['r_duration']),
            freq_av=float(evt['freq_av']),
            freq_pk=float(evt['freq_pk']),  # spikes have 0 frec and power but
            # they aren't used in ml so we just leave a None there
            power_av=None if evt['power_av'] == 0 else mt.log10(float(evt[
                                                                          'power_av'])),
            power_pk=None if evt['power_pk'] == 0 else mt.log10(float(evt[
                                                                          'power_pk'])),
            loc1=loc1,  # electrode.
            loc2=loc2,
            loc3=loc3,
            loc4=loc4,
            loc5=loc5,
            age=patient.age
        )

        if decode_type_name(evt['type']) in ['RonO', 'Fast RonO']:
            # Compass sample --> 37% has slow == [] or None, map to None
            parse_freq(info, 'slow', evt)
            parse_freq(info, 'delta', evt)
            parse_freq(info, 'theta', evt)
            parse_freq(info, 'spindle', evt)
        else:
            parse_freq(info, 'spike', evt)

        event = Event(info)
        electrode.add(event)

    # Removes electrodes with None coords or empty loc if requested by parameter
    # Notice that here elec fields are ok here after parsing events, soz has been 'or' and if a coord or loc is None/empty all their events too
    patient_ids_to_remove = set()
    for patient in patients.values():
        if remove_xyz_null:
            remove_elec_xyz_null(patient)
            if len(patient.electrodes) == 0:  # If all electrodes were removed
                patient_ids_to_remove.add(patient.id)
            check_elec_xyz_not_null(patient)
        if remove_loc_empty:
            remove_elec_loc_empty(patient)
            if len(patient.electrodes) == 0:  # If all electrodes were removed
                patient_ids_to_remove.add(patient.id)
            check_elec_loc_not_empty(patient)
    for p_name in patient_ids_to_remove:
        patients.pop(p_name)

    # Fixs EVENT inconsistencies to their electrodes
    # We know that here all electrodes have their fields ok
    evts_with_wrong_empty_loc = dict()
    for patient in patients.values():
        fix_elec_evt_soz_consistency(patient)
        fix_elec_evt_xyz_consistency(patient)
        fix_elec_evt_loc_consistency(patient, evts_with_wrong_empty_loc)

        # Just to check that the above was well done, else raise errors
        check_elec_evt_soz_consistency(patient)
        check_elec_evt_xyz_consistency(patient)
        check_elec_evt_loc_consistency(patient)

    # print('Loc empty in events from electrodes with known loc. Counts by location')
    # for k, v in evts_with_wrong_empty_loc.items():
    #    print(k, ':', v)


####################### AUXILIARY FUNCTIONS     ##########################################

# SOZ FIELD
def fix_elec_evt_soz_consistency(patient):
    # The only error to fix after the parsing of the events
    # is when elec.soz == True and event.soz == 'empty'
    for e_soz in [e for e in patient.electrodes if e.soz]:
        for type in e_soz.events.keys():
            for evt in e_soz.events[type]:
                if not evt.info['soz']:
                    evt.info['soz'] = True


def check_elec_evt_soz_consistency(patient):
    for e in patient.electrodes:
        for type in e.events.keys():
            for evt in e.events[type]:
                if evt.info['soz'] != e.soz:
                    raise RuntimeError(
                        'Found soz inconsistency in electrode/event')


# X Y Z COORDINATES
def fix_elec_evt_xyz_consistency(patient):
    # The only error to fix after ther parsing of the events
    # is when elec.x is not None and event.x is None
    for e in patient.electrodes:
        for type in e.events.keys():
            for evt in e.events[type]:
                for attr in ['x', 'y', 'z']:
                    if (getattr(e, attr) is not None) and (
                            evt.info[attr] is None):
                        evt.info[attr] = getattr(e, attr)


def check_elec_evt_xyz_consistency(patient):
    for e in patient.electrodes:
        for type in e.events.keys():
            for evt in e.events[type]:
                for attr in ['x', 'y', 'z']:
                    if getattr(e, attr) != evt.info[attr]:
                        raise RuntimeError(
                            'Found x,y,z inconsistency in electrode/event')


def remove_elec_xyz_null(patient):
    # Precondition: elec.coord is None only if all it's events have None coords
    to_remove = []
    for elec in patient.electrodes:
        if any([getattr(elec, attr) is None for attr in ['x', 'y', 'z']]):
            to_remove.append(elec)
    patient.remove_electrodes(to_remove)


def check_elec_xyz_not_null(patient):
    for elec in patient.electrodes:
        assert (
            all([getattr(elec, attr) is not None for attr in ['x', 'y', 'z']]))
        # if any([getattr(elec, attr) is None for attr in ['x', 'y', 'z']]):
        #    elec.print()


# LOCATION
def fix_elec_evt_loc_consistency(patient, evts_with_wrong_empty_loc):
    for elec in patient.electrodes:
        for type in elec.events.keys():
            for evt in elec.events[type]:
                for attr in ['loc2', 'loc3', 'loc5']:
                    # case evt loc empty elec loc not empty
                    if getattr(elec, attr) != 'empty' and evt.info[
                        attr] == 'empty':
                        # Fix inconsistency
                        evt.info[attr] = getattr(elec, attr)
                        # Log inconsistency
                        if getattr(elec,
                                   attr) in evts_with_wrong_empty_loc.keys():
                            evts_with_wrong_empty_loc[getattr(elec, attr)] += 1
                        else:
                            evts_with_wrong_empty_loc[getattr(elec, attr)] = 1
                    # case both not 'empty' but different because elec == empty evt != empty has been fixed when parsing the evt
                    elif getattr(elec, attr) != evt.info[attr] and evt.info[
                        attr] != 'empty':
                        raise RuntimeError(
                            'loc not empty disagreement among events for the same (pat_id, electrode)')
                        # log(msg=('Warning, loc5 disagreement among events for the same (patient_id, electrode)'),
                        #    msg_type='LOC5',
                        #    patient=patient.id,
                        #    electrode=elec.name
                        #    )
                        # if loc5 == 'Hippocampus':
                        #   setattr(e, attr, 'Hippocampus')


def check_elec_evt_loc_consistency(patient):
    for elec in patient.electrodes:
        for type in elec.events.keys():
            for evt in elec.events[type]:
                for attr in ['loc2', 'loc3', 'loc5']:
                    if getattr(elec, attr) != evt.info[attr]:
                        print('electrode: {0}'.format(getattr(elec, attr)))
                        print('event: {0}'.format(evt.info[attr]))
                        raise RuntimeError(
                            'Found loc inconsistency in electrode/event')


def remove_elec_loc_empty(patient):
    to_remove = []
    for elec in patient.electrodes:
        if any([getattr(elec, attr) == 'empty' for attr in
                ['loc2', 'loc3', 'loc5']]):
            to_remove.append(elec)
    patient.remove_electrodes(to_remove)


def check_elec_loc_not_empty(patient):
    for elec in patient.electrodes:
        assert (all([getattr(elec, attr) != 'empty' for attr in
                     ['loc2', 'loc3', 'loc5']]))

import math as mt

from classes import Patient, Electrode, Event
from config import EVENT_TYPES, non_intraop_patients, intraop_patients, electrodes_query_fields, \
    hfo_query_fields
from utils import log

import pymongo
from pymongo import MongoClient

class Database(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")

    def get_collections(self):
        connection = self.get_connection()
        db = connection.deckard_new

        electrodes_collection = db.Electrodes
        electrodes_collection.create_index([('patient_id', "hashed")])
        electrodes_collection.create_index([('electrode', 1)])
        electrodes_collection.create_index([('type', "hashed")])
        electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')

        event_collection = db.HFOs
        event_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')
        event_collection.create_index([('patient_id', 1), ('electrode', 1), ('intraop', 1), ('type', 1)])
        return electrodes_collection, event_collection


# Creates a patients loc dictionary for each location as parameter
def load_patients(electrodes_collection, evt_collection, intraop, loc_granularity, locations, event_type_names, models_to_run, subtypes=None, allow_null_coords=True):
    print('Loading patients...')
    loc, locations = get_locations(loc_granularity, locations)
    print('Locations: {0}'.format(locations))
    print('Event type names: {0}'.format(event_type_names))
    patients_by_loc = dict()
    for loc_name in locations:
        print('\nLocation: {0}'.format(loc_name))
        elec_filter, evt_filter = query_filters(intraop, event_type_names, loc, loc_name,
                                                hfo_subtypes=subtypes, allow_null_coords=allow_null_coords)
        print('Printing filters')
        print(elec_filter)
        print(evt_filter)
        elec_cursor = electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = evt_collection.find(evt_filter, projection=hfo_query_fields)
        patients_by_loc[loc_name] = parse_patients(elec_cursor, hfo_cursor, models_to_run)

    return patients_by_loc

# Creates the pymongo filters
def query_filters(intraop, event_type_names, loc, loc_name, hfo_subtypes=None, allow_null_coords=True):
    patient_id_intraop_cond = {'$in': intraop_patients if intraop else non_intraop_patients}
    encoded_intraop = str(int(intraop)) #int(True) == 1

    elec_filter = {'patient_id': patient_id_intraop_cond}  # 'x' : {'$ne':'-1'}, 'y' : {'$ne':'-1'}, 'z' : {'$ne':'-1'}
    evt_filter = {'patient_id': patient_id_intraop_cond, 'intraop': encoded_intraop, #probably this intraop is actually not needed but doesnt harm anyway
                  'type': {'$in': [encode_type_name(e) for e in
                                   event_type_names]}, #order doesnt matter
                  }  #   '$or': [{'type':'1'}, {'type':'2'}, {'type':'4'}, {'type':'5'} ]

    if hfo_subtypes is not None: #This says that it belongs to a subtype of a type in evt filter or the event is a Spike that doesnt have subtypes
        evt_filter['$or'] = [{'$or': [{subtype: 1} for subtype in hfo_subtypes]},
                             {'type': {'$in': ['Spikes', 'Sharp Spikes']}}]

    if not allow_null_coords:
        elec_filter['x'] = {"$ne": "-1"}
        elec_filter['y'] = {"$ne": "-1"}
        elec_filter['z'] = {"$ne": "-1"}
        evt_filter['x'] = {"$ne": "-1"}
        evt_filter['y'] = {"$ne": "-1"}
        evt_filter['z'] = {"$ne": "-1"}

    if loc is not None:
        elec_filter[loc] = loc_name
        evt_filter[loc] = loc_name
    else:
        empty_allowed = True #TODO this is to allow or not empty loc in whole brain queries
        if not empty_allowed:
            elec_filter['loc2'] = {"$ne": "empty"}
            elec_filter['loc3'] = {"$ne": "empty"}
            elec_filter['loc5'] = {"$ne": "empty"}
            evt_filter['loc2'] = {"$ne": "empty"}
            evt_filter['loc3'] = {"$ne": "empty"}
            evt_filter['loc5'] = {"$ne": "empty"}

    return elec_filter, evt_filter

# Returns a dictionary of patients with electrodes and events loaded from the cursors and solving inconsistencies
def parse_patients(electrodes_cursor, hfo_cursor, models_to_run):
    print('Parsing db patients...')
    patients_dic = dict()
    parse_electrodes(patients_dic, electrodes_cursor)
    parse_events(patients_dic, hfo_cursor, models_to_run)
    return patients_dic

# Populates patients dic with electrode entries (From Electrodes collection)
# Modifies patients
def parse_electrodes(patients, elec_cursor):
    for e in elec_cursor:
        # Patient level
        patient_id = e['patient_id']
        if not patient_id in patients.keys():
            patient = Patient(id=patient_id, age=parse_age(e))
            patients[patient_id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(e)
            if age != patient.age:
                log('Warning, age should agree among electrodes of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

        # Electrode level
        e_name = parse_elec_name(e)
        file_block = int(e['file_block'])
        x,y,z = parse_coord(e['x']), parse_coord(e['y']), parse_coord(e['z'])
        loc1, loc2, loc3, loc4, loc5 = parse_loc(e, 1), parse_loc(e, 2), parse_loc(e, 3), parse_loc(e, 4), parse_loc(e, 5)

        if not e_name in patient.electrode_names(): #First time I see this electrode for
            electrode = Electrode(name=e_name, soz=parse_soz(e['soz']),
                                  blocks={file_block: None} , x=x, y=y, z=z, soz_sc=(e['soz_sc']),
                                  loc1=loc1, loc2=loc2, loc3=loc3, loc4=loc4, loc5=loc5)
            patient.add_electrode(electrode)
        else: #The electrode exists probably because there are many blocks for the electrodes
            electrode = next(e2 for e2 in patient.electrodes if e2.name == e_name) #look up the electrode by name
            # Check consistency

            if parse_soz(e['soz']) != electrode.soz:
                log(msg=('Warning, soz disagreement among blocks of '
                         'the same patient_id, electrode, '
                         'running OR between values'),
                    msg_type='elec_blocks_SOZ_conflict',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                electrode.soz = electrode.soz or parse_soz(e['soz'])
            electrode.soz_sc = electrode.soz_sc or parse_soz(e['soz_sc'])

            if file_block not in electrode.blocks.keys():
                electrode.blocks[file_block] = None #here we will save the time of the block that is given in the HFO db...
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
                raise RuntimeError('X disagreement (both not null) in electrode blocks')

            if loc5 != 'empty' and electrode.loc5 != loc5:
                RuntimeError('This shouldnt happen, says that blocks of the same electrode have dif names none empty')
                log(msg=('Warning, loc5 disagreement among blocks for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':  # Priority
                    electrode.loc5 = loc5

# Populates patients dic with event entries (From HFO collection)
# Modifies patients
def parse_events(patients, event_collection, models_to_run):
    for evt in event_collection:
        # Patient level
        patient_id = evt['patient_id']
        if patient_id not in patients.keys():
            #raise RuntimeWarning('The parse_electrodes may should have had the patient created first')
            patient = Patient(id=patient_id, age=parse_age(evt)) #optionally you could create it in this moment
            patients[patient.id] = patient
        else:
            patient = patients[patient_id]
            # Check consistency of patient attributes
            age = parse_age(evt)
            if age != patient.age:
                log('Warning, age should be consistent among events of the same patient')
                if patient.age != 0 or age != 0:
                    patient.age = max(age, patient.age)

        assert(patient is patients[patient_id]) #They refer to the same object, but we use the short one
        # Electrode level
        e_name = parse_elec_name(evt)
        soz = parse_soz(evt['soz'])
        file_block = int(evt['file_block'])
        block_duration = float(evt['r_duration'])
        x, y, z = parse_coord(evt['x']), parse_coord(evt['y']), parse_coord(evt['z'])
        loc1, loc2, loc3, loc4, loc5 = parse_loc(evt, 1), parse_loc(evt, 2), parse_loc(evt, 3), parse_loc(evt, 4), parse_loc(evt, 5)

        if not e_name in patient.electrode_names():
            #raise RuntimeWarning('The parse_electrodes should have created the electrode for this event')
            #The code below would create a new a electrode from the HFO info
            electrode = Electrode(name= e_name, soz=soz, blocks={file_block: block_duration},
                                  x= x, y= y, z= z, soz_sc= parse_soz(evt['soz_sc']),
                                  loc1=loc1, loc2=loc2, loc3=loc3, loc4=loc4, loc5=loc5)
            patient.add_electrode(electrode)
        else:
            electrode = next(e for e in patient.electrodes if e.name == e_name)

            # Check consistency

            # SOZ
            if soz != electrode.soz:
                log(msg=('Warning, soz disagreement among event and electrode'
                         'running OR between values'),
                    msg_type='elec_evt_SOZ_conflict',
                    patient=patient.id,
                    electrode=electrode.name
                    )
            electrode.soz = electrode.soz or soz #Fixs db bug of event nsoz and electrode.soz
            electrode.soz_sc = electrode.soz_sc or parse_soz(evt['soz_sc'])

            # File block and duration
            if file_block not in electrode.blocks.keys() or electrode.blocks[file_block] is None:
                electrode.blocks[file_block] = block_duration
            else:
                if(block_duration != electrode.blocks[file_block]):
                    raise NotImplementedError('Implement which duration must be saved if they differ.')

            #X Y Z
            if electrode.x is None:
                electrode.x = x
            if electrode.y is None:
                electrode.y = y
            if electrode.z is None:
                electrode.z = z
            if x is not None and electrode.x != x:
                raise RuntimeError('X disagreement (both not null) in event/electrode')

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

            if loc5 != 'empty' and electrode.loc5 != loc5:
                log(msg=('Warning, loc5 disagreement among events for the same (patient_id, electrode)'),
                    msg_type='LOC5',
                    patient=patient.id,
                    electrode=electrode.name
                    )
                if loc5 == 'Hippocampus':
                    electrode.loc5 = loc5
        #Elec count update
        evt_type = decode_type_name(evt['type'])
        if file_block not in electrode.evt_count[evt_type].keys():
            electrode.evt_count[evt_type][file_block] = 1
        else:
            electrode.evt_count[evt_type][file_block] += 1

        # HFO_level
        info = dict(
            x=parse_coord(evt['x']),
            y=parse_coord(evt['y']),
            z=parse_coord(evt['z']),
            prediction={m: 0 for m in models_to_run},  # for saving results of a model
            proba={m: 0 for m in models_to_run},  # for saving results of a model
            soz=soz,
            type=decode_type_name(evt['type']),
            file_block=int(evt['file_block']),
            start_t=float(evt['start_t']),
            finish_t=float(evt['finish_t']),
            duration=float(evt['duration']) * 1000, #duration to milliseconds
            fr_duration=float(evt['fr_duration']),
            r_duration=float(evt['r_duration']),
            freq_av=float(evt['freq_av']),
            freq_pk=float(evt['freq_pk']),
            power_av=mt.log10(float(evt['power_av'])) if float(evt['power_av']) != 0 else 0.0, #ASK not 0
            power_pk=mt.log10(float(evt['power_pk'])) if float(evt['power_pk']) != 0 else 0.0,
            loc1=electrode.loc1,
            loc2=electrode.loc2,
            loc3=electrode.loc3,
            loc4=electrode.loc4,
            loc5=electrode.loc5,
            age=patient.age
        )
        if decode_type_name(evt['type']) in ['RonO', 'Fast RonO']:
            info['slow'] = bool(evt['slow'])
            info['slow_vs'] = 0.0 if (isinstance(evt['slow_vs'], list) or evt['slow_vs'] is None) else float(evt['slow_vs'])
            info['slow_angle'] = 0.0 if (isinstance(evt['slow_angle'], list) or evt['slow_angle'] is None) else float(evt['slow_angle'])
            info['delta'] = bool(evt['delta'])
            info['delta_vs'] = 0.0 if (isinstance(evt['delta_vs'], list) or evt['delta_vs'] is None) else float(evt['delta_vs'])
            info['delta_angle'] = 0.0 if (isinstance(evt['delta_angle'], list) or evt['delta_angle'] is None) else float(evt['delta_angle'])
            info['theta'] = bool(evt['theta'])
            info['theta_vs'] = 0.0 if (isinstance(evt['theta_vs'], list) or evt['theta_vs'] is None) else float(evt['theta_vs'])
            info['theta_angle'] = 0.0 if (isinstance(evt['theta_angle'], list) or evt['theta_angle'] is None) else float(evt['theta_angle'])
            info['spindle'] = bool(evt['spindle'])
            info['spindle_vs'] = 0.0 if (isinstance(evt['spindle_vs'], list) or evt['spindle_vs'] is None) else float(evt['spindle_vs'])
            info['spindle_angle'] = 0.0 if (isinstance(evt['spindle_angle'], list) or evt['spindle_angle'] is None) else float(evt['spindle_angle'])
        else:
            info['spike'] = bool(evt['spike'])
            info['spike_vs'] = 0.0 if (isinstance(evt['spike_vs'], list) or evt['spike_vs'] is None) else float(evt['spike_vs'])
            info['spike_angle'] = 0.0 if (isinstance(evt['spike_angle'], list) or evt['spike_angle'] is None) else float(evt['spike_angle'])

        event = Event(info)
        electrode.add(event)

    # Fixs DB bug of events with nSOZ label when electrode is SOZ
    # We know that here all electrodes have their soz labels ok, we only need to check for nsoz
    # events in soz elec. If nsoz had an error it would be also soz and that cant happen cause
    # we are assuming that electrodes are well soz tagged
    for patient in patients.values():
        #Igualar todos los soz de los evt a los de los electrodos
        for e_soz in [e for e in patient.electrodes if e.soz]:
            for type in e_soz.events.keys():
                for evt in e_soz.events[type]:
                    if not evt.info['soz']:
                        evt.info['soz'] = True

        #Igualar todos los x, y, z de los evt a los de sus respectivos electrodos
        for e in patient.electrodes:
            for type in e.events.keys():
                for evt in e.events[type]:
                    if evt.info['x'] is not None and e.x is None:
                        raise RuntimeError('Electrode None/Evt not None x,y,z inconsistencies'
                                           ' should have been handled before this point.')

                    if evt.info['x'] is None and e.x is not None:
                        evt.info['x'] = e.x
                    if evt.info['y'] is None and e.y is not None:
                        evt.info['y'] = e.y
                    if evt.info['z'] is None and e.z is not None:
                        evt.info['z'] = e.z
        #Igualar todos los loc de los evt a los de sus respesctivos electrodos
        '''
        for e in patient.electrodes:
            if any([True if getattr(e, 'loc{0}'.format(i)) == 'empty' else False for i in [2,3,5]]):
                'Printing empty loc info' # if there are empty loc electrode y print their info
                patient.print()
                electrode.print()
        '''

        ############ AUXILIARY FUNCTIONS


########################## PARSING OF FIELDS OF THE DB ####################################

# patient_id: All strings with the name of the exam. Example: '444.edf' No parsing needed

#TODO change default 0 to None and remove if we use age as feature
def parse_age(doc): #AGE 0 represents empty
    return 0.0 if doc['age'] == "empty" else float(doc['age'])

# electrode
def parse_elec_name(doc):
    if isinstance(doc['electrode'], list): #Its always length of 1 the array at list a sample but just in case
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise ValueError('Unknown type for electrode name')
    if e_name is None: #Change for RuntimeWarning if fails... in my opinion it shouldn't happen
        raise ValueError('Channel without name, it will be named with events info or stay None')

    return e_name

# soz
def parse_soz(db_representation_str): #SOZ is True if soz = "1" in db
    return (True if db_representation_str == "1" else False)

# file block : strings from 1 to 6 meaning what block of 10 minutes relative to the start of the exam, not parsing needed

# coordinates x y z
# Segun la muestra de compass no hay listas en este campo pero creo que habia saltado error asi que pongo el valor de null en la db
def parse_coord(param): # -1 Represents empty, consider filtering
    return None if (isinstance(param, list) or param == '-1') else float(param)

# locations in MNI space loc1, loc2, loc3, loc4, loc5
def all_loc_names(granularity):
    if granularity == 0:
        return ['Whole brain']
    elif granularity == 1:
        return ['Right Cerebrum', 'Left Cerebrum']
    elif granularity == 2:
        return ['Limbic Lobe', 'Parietal Lobe', 'Temporal Lobe', 'Frontal Lobe', 'Occipital Lobe']  #
    elif granularity == 3:
        return ['Fusiform Gyrus', 'Parahippocampal Gyrus',
                'Middle Temporal Gyrus', 'Postcentral Gyrus',
                'Superior Frontal Gyrus', 'Inferior Frontal Gyrus', 'Middle Frontal Gyrus']
    elif granularity == 4:
        return ['Gray Matter', 'White Matter']
    elif granularity == 5:
        return ['Hippocampus', 'Amygdala', 'Brodmann area 21', 'Brodmann area 28',
                'Brodmann area 34', 'Brodmann area 35', 'Brodmann area 36', 'Brodmann area 37']

def get_granularity(loc):
    for i in range(6):
        if loc in all_loc_names(i):
            return i
    raise RuntimeError('Unknown location name')

# Returns the loc field and list of locations for the parameters
def get_locations(loc_granularity, locations='all'):
    if loc_granularity == 0:
        loc = None
        locations = ['Whole brain']
    else:
        loc = 'loc{i}'.format(i=loc_granularity)
        locations = all_loc_names(granularity=loc_granularity) if locations == 'all' else locations
    return loc, locations

#TODO test this and change code
def parse_loc(doc, i):
    #Posible doc inputs : ['empty'], ['CH NAME'], [],, [[]], [''], 'empty', 'CH name', [[['empty']]]
    #['empty'], 'empty', [], ['']  --> 'empty'
    #['CH NAME'] 'CH NAME' --> 'CH NAME'

    if isinstance(doc['loc{0}'.format(i)], str): #Is string
        if len(doc['loc{0}'.format(i)]) > 0: #Case = 'empty' or 'CH NAME'
            loc = doc['loc{0}'.format(i)]
        else:  #Case = ''
            loc = 'empty'

    elif isinstance(doc['loc{0}'.format(i)], list): #Is a List
        if len(doc['loc{0}'.format(i)]) > 0: #Case = ['empty'], ['CH NAME'], [''], [list]
            if isinstance(doc['loc{0}'.format(i)][0], str): # Case = ['empty'], ['CH NAME'], ['']
                if len(doc['loc{0}'.format(i)][0]) > 0:
                    loc = doc['loc{0}'.format(i)][0]
                else:  # ['']
                    loc = 'empty'
            elif isinstance(doc['loc{0}'.format(i)][0], list): #Case = [list]
                if len(doc['loc{0}'.format(i)][0]) == 0: #Case # [[]]
                    loc = 'empty'
                else:
                    raise RuntimeError('Unknown type for loc')
        else: # []
            loc = 'empty'
    else:
        raise RuntimeError('Unknown type for loc')

    assert (isinstance(loc, str)) # 'empty ' represents location null
    return loc

# type of the event
def encode_type_name(name):  # Returns the type code number in the db for the event name from string
    return str(EVENT_TYPES.index(name) + 1)

def decode_type_name(type_id):  # Returnes the event name string from the db code number
    return EVENT_TYPES[int(type_id) - 1]

def all_subtype_names(hfo_type_name):  # For HFOs to get PAC
    if hfo_type_name in ['RonO', 'Fast RonO']:
        subtypes = ['slow', 'delta', 'theta', 'spindle']
    else:
        subtypes = ['spike']
    return subtypes

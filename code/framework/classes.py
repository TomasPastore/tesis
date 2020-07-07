
# Classes
#TODO split module
from config import EVENT_TYPES, HFO_TYPES, models_to_run


class Patient():
    def __init__(self, id, age):
        self.id = id
        self.age = age
        self.electrodes = []

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def remove_electrodes(self, to_remove):
        for e in to_remove:
            self.electrodes.remove(e)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

    def print(self):
        print('Printing patient {0}: '.format(self.id))
        print('\tAge: {0}'.format(self.age))
        print('\tElectrodes: ------------------------------')
        for e in self.electrodes:
            e.print()
        print('------------------------------------------')

    #Esta sirve para hacer particiones balanceadas, porque minimizo el balanceo de grupos de pacientes.
    #No me importa el balance por paciente sino por zona
    #TODO review or remove
    def get_class_balance(self, hfo_type_name):
        negative_class_count = 0
        positive_class_count = 0
        tot_count=0
        for e in self.electrodes:
            for h in e.events[hfo_type_name]:
                tot_count +=1
                if h.info['soz']:
                    positive_class_count +=1
                else:
                    negative_class_count +=1
        return negative_class_count, positive_class_count, tot_count

    # Determines if the patient has any electrode in loc
    def has_elec_in(self, loc):
        from db_parsing import get_granularity
        return any([getattr(e, 'loc{i}'.format(i=get_granularity(loc)))
                    == loc for e in self.electrodes])

    # Returns true iff the electrode has soz activity in loc_name
    def has_epilepsy_in_loc(self, granularity, loc_name):
        if granularity == 0:  # Has epilepsy in any part of the brain
            return any([e.soz for e in self.electrodes])  # assumes that e.soz is already parsed
        else: # looks if any soz electrode matches its loc_name in the correct granularity tags
            return any([e.soz and getattr(e, e.loc_field_by_granularity(granularity)) == loc_name for e in self.electrodes])

    def has_epilepsy_in_all_locs(self, granularity, locations):
        return all([self.has_epilepsy_in_loc(granularity, location) for location in locations])

    # Iff all electrodes are loc field is inside the list allowed, given as parameter locations
    # "empty" in locations allows null elements
    def has_epilepsy_restricted_to(self, granularity, locations):
        if granularity == 0 :
            return True
        else:
            restricted = True
            for e in self.electrodes:
                if e.soz and getattr(e, e.loc_field_by_granularity(granularity)) not in locations:
                    restricted = False
                    break
            return restricted


class Electrode():

    def __init__(self, name, soz, blocks, x, y, z,
                 soz_sc=None, events=None, loc1='empty', loc2='empty', loc3='empty', loc4='empty', loc5='empty', event_type_names=EVENT_TYPES):
        if events is None:
            events = {type: [] for type in event_type_names}
        self.name = name
        self.soz = soz
        self.soz_sc = soz_sc
        self.blocks = blocks
        self.x = x
        self.y = y
        self.z = z
        self.loc1 = loc1
        self.loc2 = loc2
        self.loc3 = loc3
        self.loc4 = loc4
        self.loc5 = loc5
        self.events = events
        self.evt_count = {type: {} for type in EVENT_TYPES} #TODO METHOD

    def add(self, event):
        self.events[event.info['type']].append(event)

    # Gives you the event rate per minute considering events iff it is of any type of the ones listed
    # in event_types
    def get_events_rate(self, event_types=EVENT_TYPES):
        block_rates = {block_id:[0, duration] for block_id, duration in self.blocks.items()}
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                block_rates[block][0] += count

        # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [(rate[0]/(rate[1]/60)) if rate[1] is not None else 0.0 for rate in block_rates.values()]
        block_rates_arr.sort() #avoids num errors
        return sum(block_rates_arr)/len(self.blocks)

    def get_events_count(self, event_types=EVENT_TYPES):
        result = 0
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                result += count
        return result

    def flush_cache(self, event_types):
        for event_type in event_types:
            for block in self.evt_count[event_type].keys():
                self.evt_count[event_type][block] = 0
            for evt in self.events[event_type]:
                self.evt_count[event_type][evt.info['file_block']] +=1

    def loc_field_by_granularity(self, granularity):
        return 'loc{i}'.format(i=granularity)

    def print(self):
        print('\t\tPrinting electrode {0}'.format(self.name))
        print('\t\t\tFile_blocks: {0}'.format(self.blocks))
        print('\t\t\tSOZ: {0}'.format(self.soz))
        print('\t\t\tCoords: {x}, {y} , {z}'.format(x=self.x, y=self.y, z=self.z))

        print('\t\t\tLoc5: {0}'.format(self.loc5))
        #print('\t\t\tEvents: {0}'.format([{type_name: len(l) for type_name, l in self.events.items()}]))


class Event():
    def __init__(self, info):
        self.info = info

    def reset_preds(self, models_to_run):
        self.info['prediction'] = {m:[0, 0] for m in models_to_run}
        self.info['proba'] = {m:0 for m in models_to_run}


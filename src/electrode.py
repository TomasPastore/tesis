from random import choices


class Electrode():

    def __init__(self, name, soz, blocks, x, y, z,
                 soz_sc=None, events=None, loc1='empty', loc2='empty',
                 loc3='empty', loc4='empty', loc5='empty',
                 event_type_names=None):
        if event_type_names is None:
            event_type_names = ['RonO', 'RonS', 'Spikes', 'Fast RonO',
                                'Fast RonS', 'Sharp Spikes']
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
        self.evt_count = {type: {} for type in event_type_names}

    def add(self, event):
        self.events[event.info['type']].append(event)

    # Gives you the event rate per minute considering events iff it is of any type of the ones listed
    # in event_types
    def get_events_rate(self, event_types=None):
        if event_types is None:
            event_types = ['RonO', 'RonS', 'Spikes', 'Fast RonO',
                           'Fast RonS', 'Sharp Spikes']
        block_rates = {block_id: [0, duration] for block_id, duration in
                       self.blocks.items()}
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                block_rates[block][0] += count

        # Note: rate[1] is duration, may be None if no hfo was registered for that block
        block_rates_arr = [
            (rate[0] / (rate[1] / 60)) if rate[1] is not None else 0.0 for rate
            in block_rates.values()]
        block_rates_arr.sort()  # avoids num errors
        return sum(block_rates_arr) / len(self.blocks)

    def get_events_count(self, event_types=None):
        if event_types is None:
            event_types = ['RonO', 'RonS', 'Spikes', 'Fast RonO',
                           'Fast RonS', 'Sharp Spikes']
        result = 0
        for event_type in event_types:
            for block, count in self.evt_count[event_type].items():
                result += count
        return result

    # Warning call to flush_cache after removing from electrodes
    def remove_rand_evt(self, hfo_type, art_radius=20):
        if hfo_type == 'Fast RonO':
            artifact_freq = 300  # Hz
            candidates_idx = []
            for i, evt in enumerate(self.events[hfo_type], start=0):
                if (artifact_freq - art_radius) <= evt.info['freq_av'] and \
                        evt.info['freq_av'] <= (artifact_freq + art_radius):
                    candidates_idx.append(i)
            assert (len(candidates_idx) > 0)
            idx_to_rmv = choices(candidates_idx, k=1)[0]
            self.events[hfo_type].pop(idx_to_rmv)
        else:
            raise NotImplementedError('HFO type not implemented')

    def flush_cache(self, event_types):
        for event_type in event_types:
            for block in self.evt_count[event_type].keys():
                self.evt_count[event_type][block] = 0
            for evt in self.events[event_type]:
                self.evt_count[event_type][evt.info['file_block']] += 1

    def loc_field_by_granularity(self, granularity):
        return 'loc{i}'.format(i=granularity)

    def print(self):
        print('\t\tPrinting electrode {0}'.format(self.name))
        print('\t\t\tFile_blocks: {0}'.format(self.blocks))
        print('\t\t\tSOZ: {0}'.format(self.soz))
        print(
            '\t\t\tCoords: {x}, {y} , {z}'.format(x=self.x, y=self.y, z=self.z))

        print('\t\t\tLoc5: {0}'.format(self.loc5))
        # print('\t\t\tEvents: {0}'.format([{type_name: len(l) for type_name, l in self.events.items()}]))

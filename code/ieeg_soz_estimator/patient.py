import copy



class Patient():
    def __init__(self, id, age, electrodes=None):
        if electrodes is None:
            electrodes = []
        self.id = id
        self.age = age
        self.electrodes = electrodes

    def add_electrode(self, electrode):
        self.electrodes.append(electrode)

    def remove_electrodes(self, to_remove):
        for e in to_remove:
            self.electrodes.remove(e)

    def electrode_names(self):
        return [e.name for e in self.electrodes]

    def get_electrode(self, e_name):
        try:
            return [e for e in self.electrodes if e.name==e_name][0]
        except IndexError:
            raise IndexError('Electrode not present in patient')

    def print(self):
        print('Printing patient {0}: '.format(self.id))
        print('\tAge: {0}'.format(self.age))
        print('\tElectrodes: ------------------------------')
        for e in self.electrodes:
            e.print()
        print('------------------------------------------')

    # Esta sirve para hacer particiones balanceadas,
    # minimizo el desbalanceo entre grupos de pacientes.
    # Si hago que los sets de test esten balanceados en clases hace que el
    # entrenamiento este desbalanceado y prediga mal. Es mejor balancear
    # entrenamiento. Mejor no usar.
    def get_classes_weight(self, hfo_type_name):
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

    # Returns true iff self has soz electrode in loc_name
    def has_epilepsy_in_loc(self, loc_name):
        from db_parsing import get_granularity

        if loc_name == 'Whole Brain':  # Has epilepsy in any part of the brain
            return any([e.soz for e in self.electrodes])  # assumes that e.soz is already parsed
        else: # looks if any soz electrode matches its loc_name in the correct granularity tags
            granularity = get_granularity(loc_name)
            return any([e.soz and getattr(e, e.loc_field_by_granularity(granularity)) == loc_name for e in self.electrodes])

    def has_epilepsy_in_all_locs(self, locations):
        return all([self.has_epilepsy_in_loc(location) for location in locations])

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


# -*- coding: utf-8 -*-
# TODO review and update module

import unittest
import numpy as np
import pymongo
import math as mt
import graphics

from conf import ( ML_MODELS_TO_RUN)
from db_parsing import (Database, parse_patients, get_locations,
                        encode_type_name, EVENT_TYPES, HFO_TYPES,
                        intraop_patients, non_intraop_patients,
                        electrodes_query_fields,
                        hfo_query_fields, query_filters)
from driver import Driver


class hfoDBTest(unittest.TestCase):
    def setUp(self):
        print('Setting up test suite')
        db = Database()
        connection = db.get_connection()
        db = connection.deckard_new

        electrodes_collection = db.Electrodes
        electrodes_collection.create_index([('patient_id', "hashed")])
        electrodes_collection.create_index([('electrode', 1)])
        electrodes_collection.create_index([('type', "hashed")])
        electrodes_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')
        self.electrodes_collection = electrodes_collection
        hfo_collection = db.HFOs
        hfo_collection.create_index([('loc5', pymongo.TEXT)], default_language='english')
        hfo_collection.create_index([('patient_id', 1), ('electrode', 1), ('intraop', 1), ('type', 1)])
        self.hfo_collection = hfo_collection
        self.blocks = 4

    def test_mock_pass(self):
        blocks = 4
        self.assertEqual(blocks, self.blocks)

    def test_we_load_correct_non_intraop_info(self):
        blocks = 4
        self.assertEqual(blocks, self.blocks)

    def test_soz_agrees_in_blocks_and_events_of_same_electrode(self):
        event_type_names = EVENT_TYPES
        loc_granularity = 0
        locations = 'all'
        intraop = False
        loc, locations = get_locations(loc_granularity, locations)
        print('Locations: {0}'.format(locations))
        print('Event type names: {0}'.format(event_type_names))
        event_type_data_by_loc = dict()
        loc_name = locations[0]
        print('\nLocation: {0}'.format(loc_name))
        event_type_data_by_loc[loc_name] = dict()
        elec_filter, evt_filter = query_filters(intraop, event_type_names, loc, loc_name)
        elec_cursor = self.electrodes_collection.find(elec_filter, projection=electrodes_query_fields)
        hfo_cursor = self.hfo_collection.find(evt_filter, projection=hfo_query_fields)
        patients_dic = parse_patients(elec_cursor, hfo_cursor, ML_MODELS_TO_RUN)


        bugs_by_type = dict()
        for type in event_type_names:
            bugs_by_type[ '{type}_evt_SOZ_disagreement'.format(type=type) ] = 0

        for pat_id, patient in patients_dic.items():
            for e in patient.electrodes:
                for evt_type_name in event_type_names:
                    for evt in e.events[evt_type_name]:
                        if e.soz != evt.info['soz'] or (not evt.info['soz'] and any([e.pevt_count[t] > 0 for t in event_type_names]) ):
                            bugs_by_type['{type}_evt_SOZ_disagreement'.format(type=evt_type_name)]+= 1
        print(bugs_by_type)
        for type in event_type_names:
            self.assertEqual(0, bugs_by_type['{type}_evt_SOZ_disagreement'.format(type=type)])

    def test_experiments_are_implemented(self):
        #TODO complete figures
        exp_driver = Driver(self.electrodes_collection, self.hfo_collection)
        for number, roman_num, letter in [('1', None, None),
                                             ('2','i', None),
                                             ('2', 'ii', None)
                                        ]:
            try:
                exp_driver.run_experiment( number, roman_num, letter)
            except NotImplementedError as e:
                raise e
if __name__ == "__main__":
    unittest.main()

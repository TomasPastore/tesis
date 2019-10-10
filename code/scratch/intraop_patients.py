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

def main():
    connection = Connect.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs
    electrodes_collection = db.Electrodes

    elec_cursor = electrodes_collection.find({}, projection=['patient_id'])
    intraop_hfo_cursor = hfo_collection.find({'intraop':'1'}, projection=['patient_id', 'intraop'])
    non_intraop_hfo_cursor = hfo_collection.find({'intraop':'0'}, projection=['patient_id', 'intraop'])

    elec_patients = set()
    for e in elec_cursor:
        elec_patients.add(e['patient_id'])
    
    intraop_hfo_patients = set()
    for h in intraop_hfo_cursor:
        intraop_hfo_patients.add(h['patient_id'])

    non_intraop_hfo_patients = set()
    for h in non_intraop_hfo_cursor:
        non_intraop_hfo_patients.add(h['patient_id'])

    print('Patients in Electrodes collection:')
    print(sorted(list(elec_patients)))

    print('Intraop Patients HFO collection:')
    print(sorted(list(intraop_hfo_patients)))


    print('Non Intraop Patients HFO collection:')
    print(sorted(list(non_intraop_hfo_patients)))


    print('Uncertain patients (appear in intraop and non intraop)')
    print(intraop_hfo_patients.intersection(non_intraop_hfo_patients))

    hfo_tot_patients = intraop_hfo_patients.union(non_intraop_hfo_patients)
    print('HFO total patients count: {0}'.format(len(hfo_tot_patients)))
    print('Electrodes patients count: {0}'.format(len(elec_patients)))


    print('(Intraop U non_intraop) not found in Electrodes patients')
    print( hfo_tot_patients - elec_patients )

    print('Electrodes patients not found in (intraop U non_intraop)')
    print(elec_patients - hfo_tot_patients)

if __name__ == "__main__":
    main() 

'''
Patients in Electrodes collection:
['2061', '3162', '3444', '3452', '3656', '3748', '3759', '3799', '3853', '3900', '3910', '3943', '3967', '3997', '4002', '4009', '4013', '4017', '4028', '4036', '4041', '4047', '4048', '4050', '4052', '4060', '4061', '4066', '4073', '4076', '4077', '4084', '4085', '4089', '4093', '4099', '4100', '4104', '4110', '4116', '4122', '4124', '4145', '4150', '4163', '4166', '448', '449', '451', '453', '454', '456', '458', '462', '463', '465', '466', '467', '468', '470', '472', '473', '474', '475', '477', '478', '479', '480', '481', '729', '831', 'IO001', 'IO001io', 'IO002', 'IO002io', 'IO004', 'IO005', 'IO005io', 'IO006', 'IO006io', 'IO008', 'IO008io', 'IO009', 'IO009io', 'IO010', 'IO010io', 'IO011io', 'IO012', 'IO012io', 'IO013', 'IO013io', 'IO014', 'IO015', 'IO015io', 'IO017', 'IO017io', 'IO018', 'IO018io', 'IO019', 'IO021', 'IO021io', 'IO022', 'IO022io', 'IO023', 'IO024', 'IO025', 'IO027', 'M0423', 'M0580', 'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']
Intraop Patients HFO collection:
['IO001io', 'IO002io', 'IO005io', 'IO006io', 'IO008io', 'IO009io', 'IO010io', 'IO011io', 'IO012io', 'IO013io', 'IO015io', 'IO017', 'IO017io', 'IO018io', 'IO021io', 'IO022io', 'M0423', 'M0580', 'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']
Non Intraop Patients HFO collection:
['2061', '3162', '3444', '3452', '3656', '3748', '3759', '3799', '3853', '3900', '3910', '3943', '3967', '3997', '4002', '4009', '4013', '4017', '4028', '4036', '4041', '4047', '4048', '4050', '4052', '4060', '4061', '4066', '4073', '4076', '4077', '4084', '4085', '4089', '4093', '4099', '4100', '4104', '4110', '4116', '4122', '4124', '4145', '4150', '4163', '4166', '448', '449', '451', '453', '454', '456', '458', '462', '463', '465', '466', '467', '468', '470', '472', '473', '474', '475', '477', '478', '479', '480', '481', '729', '831', 'IO001', 'IO002', 'IO004', 'IO005', 'IO006', 'IO008', 'IO009', 'IO010', 'IO012', 'IO013', 'IO014', 'IO015', 'IO017', 'IO018', 'IO019', 'IO021', 'IO022', 'IO023', 'IO024', 'IO025', 'IO027']

'''
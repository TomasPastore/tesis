from pathlib import Path

from models import naive_bayes, random_forest, svm_m, balanced_random_forest, xgboost, sgd, simulator

DEBUG = True
LOG = {}

#PROBAR CON EL FILTRO EN ELECTRODOS DE FILTRAR LOS X Y Z VALIDOS

#Global var for biomarkers to be loaded and analized
type_names_to_run =  ['RonS'] #default value
#type_names_to_run =  ['RonO', 'RonS', 'Fast RonO', 'Fast RonS']

#Global var of the models of ml to run, XGBoost default
models_to_run = ['XGBoost'] # 'Balanced random forest''Linear SVM' 'Simulated'
models_to_run_obj = [xgboost] #The python objects of the sklearn class
models_dic = {'XGBoost': xgboost,
              'Linear SVM': svm_m,
              'Random Forest': random_forest,
              'Balanced random forest':balanced_random_forest,
              'SGD': sgd,
              'Bayes': naive_bayes,
              'Simulated': simulator
              }

EVENT_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes'] #confirmar
HFO_TYPES = ['RonO', 'RonS', 'Fast RonO', 'Fast RonS'] #confirmar
HFO_SUBTYPES = ['slow', 'delta', 'theta', 'spindle', 'spike'] #confirmar ordenadas

electrodes_query_fields = ['patient_id', 'age', 'file_block', 'electrode',
                           'loc1', 'loc2', 'loc3', 'loc4', 'loc5',
                           'soz', 'soz_sc', 'x', 'y', 'z', 'age']

hfo_query_fields = ['patient_id', 'age', 'file_block', 'electrode',
                    'loc1', 'loc2', 'loc3', 'loc4', 'loc5', 'type', 'soz', 'soz_sc',
                    'start_t', 'finish_t', 'duration', 'fr_duration', 'r_duration',
                    'freq_av', 'freq_pk', 'power_av', 'power_pk',
                    'slow', 'slow_vs', 'slow_angle',
                    'delta', 'delta_vs', 'delta_angle',
                    'theta', 'theta_vs', 'theta_angle',
                    'spindle', 'spindle_vs', 'spindle_angle',
                    'spike', 'spike_vs', 'spike_angle', 'x', 'y', 'z']

all_patient_names = ['2061', '3162', '3444', '3452', '3656', '3748', '3759', '3799', '3853', '3900', '3910', '3943',
                     '3967', '3997', '4002', '4009', '4013', '4017', '4028', '4036', '4041', '4047', '4048', '4050',
                     '4052', '4060', '4061', '4066', '4073', '4076', '4077', '4084', '4085', '4089', '4093', '4099',
                     '4100', '4104', '4110', '4116', '4122', '4124', '4145', '4150', '4163', '4166', '448', '449',
                     '451', '453', '454', '456', '458', '462', '463', '465', '466', '467', '468', '470', '472', '473',
                     '474', '475', '477', '478', '479', '480', '481', '729', '831', 'IO001', 'IO001io', 'IO002',
                     'IO002io', 'IO004', 'IO005', 'IO005io', 'IO006', 'IO006io', 'IO008', 'IO008io', 'IO009', 'IO009io',
                     'IO010', 'IO010io', 'IO011io', 'IO012', 'IO012io', 'IO013', 'IO013io', 'IO014', 'IO015', 'IO015io',
                     'IO017', 'IO017io', 'IO018', 'IO018io', 'IO019', 'IO021', 'IO021io', 'IO022', 'IO022io', 'IO023',
                     'IO024', 'IO025', 'IO027', 'M0423', 'M0580', 'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']


removed_cause_was_in_both = 'IO017'
intraop_patients = ['IO001io', 'IO002io', 'IO005io', 'IO006io', 'IO008io', 'IO009io', 'IO010io', 'IO011io', 'IO012io',
                    'IO013io', 'IO015io', 'IO017io', 'IO018io', 'IO021io', 'IO022io', 'M0423', 'M0580',
                    'M0605', 'M0761', 'M0831', 'M1056', 'M1072', 'M1264']
non_intraop_patients = ['2061', '3162', '3444', '3452', '3656', '3748', '3759', '3799', '3853', '3900', '3910', '3943',
                        '3967', '3997', '4002', '4009', '4013', '4017', '4028', '4036', '4041', '4047', '4048', '4050',
                        '4052', '4060', '4061', '4066', '4073', '4076', '4077', '4084', '4085', '4089', '4093', '4099',
                        '4100', '4104', '4110', '4116', '4122', '4124', '4145', '4150', '4163', '4166', '448', '449',
                        '451', '453', '454', '456', '458', '462', '463', '465', '466', '467', '468', '470', '472',
                        '473', '474', '475', '477', '478', '479', '480', '481', '729', '831', 'IO001', 'IO002', 'IO004',
                        'IO005', 'IO006', 'IO008', 'IO009', 'IO010', 'IO012', 'IO013', 'IO014', 'IO015',
                        'IO018', 'IO019', 'IO021', 'IO022', 'IO023', 'IO024', 'IO025', 'IO027']

# 25% of hippocampus RonS patinets query randomly selected
hip_rons_validation_names = ['3997', 'IO001', '463', '4099', '466', 'IO025', '458', 'IO027', 'IO002', '480', '454']

color_list = [ 'blue', 'green', 'magenta', 'yellow', 'lightcyan', 'black', 'mediumslateblue', 'lime', 'darkviolet', 'gold']

EXPERIMENTS_FOLDER = str(Path('~/Documentos/lic_computacion/tesis/experiments/').expanduser().resolve())

experiment_saving_path = {num:dict() for num in range(1,7)} #7 'Experiments'
experiment_saving_path[1][0] = str(Path(EXPERIMENTS_FOLDER, '1_global_data/' ))
experiment_saving_path[2][0] = str(Path(EXPERIMENTS_FOLDER, '2_rate_soz_vs_nsoz/' ))
experiment_saving_path[3][0] = str(Path(EXPERIMENTS_FOLDER, '3_predicting_soz_with_rate/' ))
experiment_saving_path[3][1] = str(Path(experiment_saving_path[3][0], 'hfos_vs_spikes/hfos_vs_spikes' ))
experiment_saving_path[3][2] = str(Path(experiment_saving_path[3][0], 'hfo_types_vs_spikes/hfo_types_vs_spikes' ))
experiment_saving_path[3][3] = str(Path(experiment_saving_path[3][0], 'hfo_types_localized/hfo_types_localized' ))
experiment_saving_path[3][4] = str(Path(experiment_saving_path[3][0], 'pse_hfo_rate_auc_relation/pse_hfo_rate_auc_relation' ))




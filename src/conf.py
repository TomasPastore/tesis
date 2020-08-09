# Global variables and definition settings
from pathlib import Path
import utils
import ml_algorithms

DEBUG = True
TEST_BEFORE_RUN = False

ML_MODELS_TO_RUN = ['XGBoost']  # Default model, more in ml_algorithms.py
# Default global var of the models of ml to run
ML_MODELS_DIC = {
    'XGBoost': ml_algorithms.xgboost,
    'Linear SVM': ml_algorithms.svm_m,
    'Random Forest': ml_algorithms.random_forest,
    'Balanced random forest': ml_algorithms.balanced_random_forest,
    'SGD': ml_algorithms.sgd,
    'Bayes': ml_algorithms.naive_bayes,
    'Simulated': ml_algorithms.simulator
}

# Path settings for reading/writing files
PROJECT_ROOT_DIR = utils.get_project_root()

FIG_FOLDER_DIR = str(Path(PROJECT_ROOT_DIR, 'figures'))
VALIDATION_NAMES_BY_LOC_PATH = str(Path(PROJECT_ROOT_DIR,
                                        'src/validation_names_by_loc.json'))
FRonO_KMEANS_EXP_DIR = str(Path(PROJECT_ROOT_DIR, 'figures/FRonO_k_means'))

# ORCA dependency is required if you want automatically save plotly figs, if the
# path doesnt exist code will skip this save, type 'which orca' in shell to get it
ORCA_EXECUTABLE = '/home/tpastore/.nvm/versions/node/v14.5.0/bin/orca'  #
# npm opt

# Paths for saving figures for each experiment defined in driver.py
# Note: untagged versions are deprecated
FIG_SAVE_PATH = dict()

# 1 Data dimensions and sleep patients
FIG_SAVE_PATH[1] = str(Path(FIG_FOLDER_DIR, '1_global_data'))  # dir

# 2 Stats of features and event rate SOZ vs NSOZ
FIG_SAVE_PATH[2] = dict()
FIG_SAVE_PATH[2]['dir'] = str(Path(FIG_FOLDER_DIR, '2_stats'))
FIG_SAVE_PATH[2]['i'] = dict()
FIG_SAVE_PATH[2]['i']['a'] = str(Path(FIG_SAVE_PATH[2]['dir'],
                                      '2_i_whole_brain_untagged_coords'))  # dir
# #DEPRECATED
FIG_SAVE_PATH[2]['i']['b'] = str(Path(FIG_SAVE_PATH[2]['dir'],
                                      '2_i_whole_brain_tagged_coords'))  # dir
FIG_SAVE_PATH[2]['ii'] = str(
    Path(FIG_SAVE_PATH[2]['dir'], '2_ii_localized'))  # dir

# 3 Event rate soz predictor baselines
FIG_SAVE_PATH[3] = dict()
FIG_SAVE_PATH[3]['dir'] = str(
    Path(FIG_FOLDER_DIR, '3_rate_soz_predictor_baselines'))
FIG_SAVE_PATH[3]['0'] = str(Path(FIG_SAVE_PATH[3]['dir'],
                                 '3_0_first_steps/hfos_and_spikes'))  # path
FIG_SAVE_PATH[3]['i'] = dict()
FIG_SAVE_PATH[3]['i']['dir'] = str(Path(FIG_SAVE_PATH[3]['dir'],
                                        '3_i_whole_brain'))
FIG_SAVE_PATH[3]['i']['a'] = str(Path(FIG_SAVE_PATH[3]['i']['dir'],
                                      '3_i_whole_brain_untagged_coords'))  # dir
FIG_SAVE_PATH[3]['i']['b'] = str(Path(FIG_SAVE_PATH[3]['i']['dir'],
                                      '3_i_whole_brain_tagged_coords'))  # dir

FIG_SAVE_PATH[3]['ii'] = dict()
FIG_SAVE_PATH[3]['ii']['dir'] = str(Path(FIG_SAVE_PATH[3]['dir'],
                                         '3_ii_pse_hfo_rate_auc_relation'))
FIG_SAVE_PATH[3]['iii'] = dict()
FIG_SAVE_PATH[3]['iii']['dir'] = str(Path(FIG_SAVE_PATH[3]['dir'],
                                          '3_iii_localized'))

# 4 ml_hfo_classfiers
FIG_SAVE_PATH[4] = dict()
FIG_SAVE_PATH[4]['dir'] = str(Path(FIG_FOLDER_DIR, '4_ml_hfo_classifiers'))
FIG_SAVE_PATH[4]['i'] = dict()
FIG_SAVE_PATH[4]['i']['a'] = str(Path(FIG_SAVE_PATH[4]['dir'],
                                      '4_i_whole_brain_coords_untagged'))  # dir
FIG_SAVE_PATH[4]['i']['b'] = str(Path(FIG_SAVE_PATH[4]['dir'],
                                      '4_i_whole_brain_coords_tagged'))  # dir
FIG_SAVE_PATH[4]['ii'] = dict()
FIG_SAVE_PATH[4]['ii']['Hippocampus'] = str(Path(FIG_SAVE_PATH[4]['dir'],
                                                 '4_ii_localized_Hippocampus'))  # dir

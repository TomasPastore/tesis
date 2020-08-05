import os  # libreria para clear screen
import \
    sys  # esto es una libreria para poder terminar el programa llamando a sys.exit(), es una opcion de menu()
import time
import warnings
from sys import version as py_version

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import TEST_BEFORE_RUN, FRonO_KMEANS_EXP_DIR
from db_parsing import Database

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
import tests
from driver import Driver
import unittest


def main(interactive_exp_menu=False):
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    exp_driver = Driver(elec_collection, evt_collection)

    # Paper Frontiers
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    # Thesis
    if interactive_exp_menu:
        experiment_menu(exp_driver)
    else:
        if TEST_BEFORE_RUN:
            unittest.main(tests, exit=False)

        # Experiment list for the driver:
        #exp_driver.run_experiment( number=2, roman_num='i', letter='b')
        FRonO_paper(elec_collection, evt_collection)


def experiment_menu(exp_driver):
    clear_screen()
    print('Experiment list:')
    print('1) Data global analysis')
    print('2) HFO rate in SOZ vs NSOZ')
    print('3) Predicting SOZ with rates: Baselines')
    print('4) ML HFO classifiers')
    print('5) HFO rate Baseline vs ml filtered pHFO rate')
    print('6) Simulator')
    print('7) Exit')
    option = int(input('Choose a number from the options above: '))

    if option == 1:
        exp_driver.run_experiment(number=1) # intraop and dimensions in
        # localized and whole brain regions
        go_to_menu_after(5, exp_driver)
    elif option == 2:
        exp_driver.run_experiment(number=2, roman_num='i', letter='b') #whole
        # brain tagged
        exp_driver.run_experiment(number=2, roman_num='ii') #localized
        go_to_menu_after(5, exp_driver)
    elif option == 3:
        exp_driver.run_experiment(number=3, roman_num='0') # whole brain HFOs
        # together
        exp_driver.run_experiment(number=3, roman_num='i', letter='a') #whole
        # brain untagged
        exp_driver.run_experiment(number=3, roman_num='i', letter='b') #whole
        # brain tagged
        exp_driver.run_experiment(number=3, roman_num='ii') # PSE AUC relation
        exp_driver.run_experiment(number=3, roman_num='iii') # localized
        go_to_menu_after(5, exp_driver)

    elif option == 4:
        # Whole brain coords untagged
        exp_driver.run_experiment(number=4, roman_num='i', letter='a')

        # Whole brain coords tagged
        exp_driver.run_experiment(number=4, roman_num='i', letter='b')

        # Localized Hippocampus
        exp_driver.run_experiment(number=4, roman_num='ii', letter='a')
        go_to_menu_after(5, exp_driver)

    elif option == 5:
        raise NotImplementedError('Future work?')
        go_to_menu_after(5, exp_driver)

    elif option == 6:
        raise NotImplementedError('Future work?')
        go_to_menu_after(5, exp_driver)

    elif option == 7:
        sys.exit()
    else:
        raise NotImplementedError('Option {0} was left as future '
                                  'work.'.format(option))

def clear_screen():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

def go_to_menu_after(seconds, exp_driver):
    seconds = 5
    while seconds > 0:
        print('Going back to menu in {0}...'.format(seconds))
        time.sleep(1)  # wait 1 sec
        seconds = seconds - 1
    experiment_menu(exp_driver)

def FRonO_paper(elec_collection, evt_collection):
    from soz_predictor import evt_rate_soz_pred_baseline_localized
    evt_rate_soz_pred_baseline_localized(elec_collection,
                                         evt_collection,
                                         intraop=False,
                                         restrict_to_tagged_coords=True,
                                         restrict_to_tagged_locs=True,
                                         evt_types_to_load=['Fast RonO'],
                                         evt_types_to_cmp=[['Fast RonO']],
                                         locations={0: ['Whole Brain']},
                                         saving_dir=FRonO_KMEANS_EXP_DIR,
                                         #return_pat_dic_by_loc=True,
                                         plot_rocs=False,
                                         remove_elec_artifacts=True)


if __name__ == "__main__":
    main(interactive_exp_menu=False)

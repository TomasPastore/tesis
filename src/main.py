import os  # library for screen cleaning in interactive mode
import sys  # library for exit() in interactive mode
import time
import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from conf import TEST_BEFORE_RUN, FRonO_KMEANS_EXP_DIR
from db_parsing import Database
import tests
from driver import Driver
import unittest
import argparse


def main(interactive_exp_menu=False):
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    exp_driver = Driver(elec_collection, evt_collection)

    # Paper Frontiers
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    # Paper FRonO
    # from FRonO_paper import scratch_steps
    # scratch_steps(elec_collection, evt_collection)

    # Thesis and main project code
    if interactive_exp_menu:
        experiment_menu(exp_driver)
    else:
        if TEST_BEFORE_RUN:
            unittest.main(tests, exit=False)

        # Call an specific driver function if not interactive mode
        exp_driver.run_experiment(number=2, roman_num='i', letter='b')

def experiment_menu(exp_driver):
    clear_screen()
    print('Experiment list:')
    print('                ')
    print('1) Data global analysis')
    print('2) Data stats analysis. Features and HFO rate in SOZ vs NSOZ.')
    print('3) Predicting SOZ with event rates: Baselines')
    print('4) ML HFO classifiers')
    print('5) HFO rate Baseline vs ML filtered pHFO rate')
    print('6) Simulator')
    print('7) Exit')
    option = int(input('Choose a number from the options above: '))

    if option == 1:
        exp_driver.run_experiment(number=1)  # intraop and dimensions in
        # localized and whole brain regions
        go_to_menu_after(5, exp_driver)
    elif option == 2:
        # Data stats analysis
        exp_driver.run_experiment(number=2, roman_num='i', letter='b')
        exp_driver.run_experiment(number=2, roman_num='ii')  # localized
        go_to_menu_after(5, exp_driver)
    elif option == 3:
        # Whole Brain simple model all HFOs vs Spikes'
        exp_driver.run_experiment(number=3, roman_num='0')

        # Whole Brain untagged (N = 91)
        # exp_driver.run_experiment(number=3, roman_num='i', letter='a')

        # Whole Brain untagged (N = 57)
        exp_driver.run_experiment(number=3, roman_num='i', letter='b')

        # PSE AUC relation
        exp_driver.run_experiment(number=3, roman_num='ii')

        # Localized Hippocampus
        exp_driver.run_experiment(number=3, roman_num='iii')

        go_to_menu_after(5, exp_driver)

    elif option == 4:
        # Whole Brain coords untagged
        exp_driver.run_experiment(number=4, roman_num='i', letter='a')

        # Whole Brain coords tagged
        exp_driver.run_experiment(number=4, roman_num='i', letter='b')

        # Localized Hippocampus
        exp_driver.run_experiment(number=4, roman_num='ii', letter='a')
        go_to_menu_after(5, exp_driver)

    elif option == 5:
        raise NotImplementedError('TODO')

    elif option == 6:
        raise NotImplementedError('TODO')

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
    while seconds > 0:
        print('Going back to menu in {0}...'.format(seconds))
        time.sleep(1)  # wait 1 sec
        seconds = seconds - 1
    experiment_menu(exp_driver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive_mode",
                        help="Run the experiments interactively.",
                        required=False,
                        default=False,
                        action='store_true',
                        )
    args = parser.parse_args()
    main(interactive_exp_menu=args.interactive_mode)

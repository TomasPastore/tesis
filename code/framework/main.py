import random
import sys
from sys import version as py_version
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from matplotlib import \
    pyplot as plt  # todo ver donde se usa, deberia estar solo en graphics
import math as mt
from datetime import timedelta
from config import (EVENT_TYPES, exp_save_path, TEST_BEFORE_RUN,
                    experiment_default_path,
                    intraop_patients, models_to_run)
from db_parsing import Database, parse_patients, get_locations, \
    get_granularity, encode_type_name, all_loc_names, load_patients, query_filters, \
    all_subtype_names
from metrics import print_metrics

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
from utils import all_subsets, LOG, print_info
import tests
from driver import Driver
import unittest
import graphics
import math as mt


def main():
    db = Database()
    elec_collection, evt_collection = db.get_collections()
    # phase_coupling_paper(hfo_collection) # Paper Frontiers

    # Thesis
    if TEST_BEFORE_RUN:
        # TODO Agregar a informe, tests
        unittest.main(tests, exit=False)

    exp_driver = Driver(elec_collection, evt_collection)
    # Experiment list for the driver:
    exp_driver.run_experiment( number=3, roman_num='0', letter='a')


if __name__ == "__main__":
    main()

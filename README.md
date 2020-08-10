
# Developed by 

**Tomás Ariel Pastore** <tpastore@dc.uba.ar> in
collaboration with *Shennan A. Weiss* and *Diego F. Slezak* at **LIAA U.B.A**.


# Summary:

The aim of this code is to serve as tool to analyse a large HFO database.  
**Are HFOs good predictors of the SOZ to help refractory epilepsy treatment?**

# Dependencies 
	Install python packages detailed in requirements.txt in project root

# Run
	From project root you can execute the following commands in shell: 
	1) Run in interactive mode command: 
	    **python src/main.py -i** 
    2) Run an specific drive function setted in main() function manually:
        **python src/main.py**  
        
# Project layout:

ieeg_soz_predictor.
├── docs
│   ├── bibliography
│   ├── DB_fields_schema
│   ├── manuscript
│   └── pastore proposal.odt
├── figures
│   ├── 1_global_data
│   └── 2_stats
├── lib \# third party code 
│   ├── matlab_circ_package
│   └── README.txt
├── README.md
├── requirements.txt
└── src
    ├── artifacts.py
    ├── conf.py                                   \# Definitions and global vars
    ├── db_parsing.py                      \# Database parsing and normalization
    ├── driver.py                       \# Abstraction to navigate the main code
    ├── electrode.py                                          \# Electrode class
    ├── event.py                                                  \# Event class
    ├── FRonO_paper.py
    ├── graphics.py                                        \# Figures generation
    ├── main.py                                 \# Program execution entry point
    ├── ml_algorithms.py            \# Sklearn machine learning algorithms calls
    ├── ml_hfo_classifier.py     \# Sklearn for classifying pHFOs using features
    ├── ml_hfo_simulator.py       \# Simulating classifier for fixed recall/prec
    ├── partition_builder.py       \# Making partitions of patients for crossval
    ├── patient.py                                              \# Patient class
    ├── review_code.py
    ├── scratch.py                          \# First queries and data dimensions
    ├── soz_predictor.py               \# Event-rate baseline per location, type
    ├── stats.py                              \# HFO rate and feature stats code
    ├── tests.py                                                   \# Test suite
    ├── utils.py                                                 \# Usefull code
    └── validation_names_by_loc.json      \# Dict with random validation patient



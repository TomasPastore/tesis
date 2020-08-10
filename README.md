
# Developed by 

**Tomás Ariel Pastore** <tpastore@dc.uba.ar> in
collaboration with *Shennan A. Weiss* and *Diego F. Slezak* at **LIAA U.B.A**.

\

# Summary:

The aim of this code is to serve as tool to analyse a large HFO database.  
** Are HFOs good predictors of the SOZ to help refractory epilepsy treatment? **

\

# Dependencies 
	Install python packages detailed in requirements.txt in project root

# Run
	From project root you can execute the following commands in shell: 
	1) Run in interactive mode command: 
	    ** python src/main.py -i ** 
    2) Run an specific drive function setted in main() function manually:
        ** python src/main.py **  
        
# Project layout:

ieeg_soz_predictor
│   README.md
|   .gitignore 
|   docs \# Bibliography, manuscript, thesis proposal
|   lib  \# third parties code, matlab_engine python package may use matlab code
|
└───src # Project code in Python  
│   │   validation_names_by_loc.json \# Dictionary with random validation patient 
|   |   conf.py                      \# Definitions and global vars
|   |   main.py                      \# Program execution entry point
|   |   db_parsing.py                \# Database parsing and normalization
|   |   patient.py                   \# Patient class
|   |   electrode.py                 \# Electrode class
|   |   event.py                     \# Event class
|   |   scartch.py                   \# First queries and data dimensions
|   |   stats.py                     \# HFO rate and feature stats code
|   |   partition_builder.py         \# Making partitions of patients for crossval
|   |   ml_algorithms.py             \# Sklearn machine learning algorithms calls
|   |   soz_predictor.py             \# Event-rate baseline per location, type
|   |   ml_hfo_classifier.py         \# Simulating classifier for fixed recall/prec
|   |   graphics.py                  \# Figures generation 
|   |   utils.py                     \# Usefull code
|   |   tests.py                     \# Test suite
|   |   driver.py                    \# Abstraction to navigate the main code 



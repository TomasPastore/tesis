# TODO review and update
# 7) Simulation of the ml predictor to understand needed performance to improve HFO rate baseline
def simulator(elec_collection, evt_collection):
    models_to_run = ['XGBoost', 'Simulated']
    for conf in [0.6, 0.7, 0.8,
                 0.9]:  # confianzas del simulador una antes y una despues de baseline
        comp_with = '{0} Simulator '.format(conf)
        print('Conf: {0}'.format(conf))
        tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        #ml_phfo_models(elec_collection, evt_collection, 'Hippocampus',
        #               'RonS',
        #               tol_fprs=tol_fprs, models_to_run=models_to_run,
        #               comp_with=comp_with, conf=conf)


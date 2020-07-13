

# 6) pHFOs rate VS HFO rate baseline
# Tambien probe en vez de usar la prop de phfos > thresh en vez de hfo rate
# only added to test a classifier calc+ug baseline rate as feature

def Hippocampal_RonS_gradual_filters(elec_collection, evt_collection):
    model_name = 'XGBoost'
    models_to_run = [model_name]
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_phfo_models(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                   tol_fprs=tol_fprs, models_to_run=models_to_run,
                   comp_with='{0} '.format(model_name))


def ml_with_rate(elec_collection, evt_collection, loc_name, hfo_type_name,
                 tol_fprs):
    intraop = False
    model_name = 'XGBoost'
    models_to_run = [model_name]
    event_type_data_by_loc = {loc_name: {}}
    loc, locations = get_locations(5, [loc_name])
    target = ['model_pat']
    elec_filter, evt_filter = query_filters(intraop, [hfo_type_name], loc,
                                            loc_name)
    elec_cursor = elec_collection.find(elec_filter,
                                       projection=electrodes_query_fields)
    hfo_cursor = evt_collection.find(evt_filter,
                                     projection=hfo_query_fields)
    patients_dic = parse_patients(elec_cursor, hfo_cursor, models_to_run)
    print('Total patients {0}'.format(len(patients_dic)))

    for p in patients_dic.values():
        for e in p.electrodes:
            rate = e.get_events_rate([hfo_type_name])
            for h in e.events[hfo_type_name]:
                h.info['rate'] = rate

    target_patients = phfo_predictor(loc_name, hfo_type_name, patients_dic,
                                     target=target, models=models_to_run)

    all_target = 'model_pat' in target and 'validation_pat' in target
    if all_target:
        name = ' baseline all'
    if not all_target:
        name = ' baseline {0}'.format(target[0])
    event_type_data_by_loc[loc_name][hfo_type_name + name] = region_info(
        {p.id: p for p in target_patients}, [hfo_type_name])

    print('Running model {0}'.format(model_name))
    labels, preds, probs = gather_folds(model_name, hfo_type_name,
                                        target_patients=target_patients)
    print('Displaying metrics for phfo classifier')
    print_metrics(model_name, hfo_type_name, labels, preds, probs)

    # SOZ HFO RATE MODEL
    fpr, tpr, thresholds = roc_curve(labels, probs)
    for tol_fpr in tol_fprs:
        thresh = get_soz_confidence_thresh(fpr, thresholds,
                                           tolerated_fpr=tol_fpr)  # if the prob is more than this thresh I will consider it for the mean
        filtered_pat_dic = phfo_thresh_filter(target_patients,
                                              hfo_type_name,
                                              thresh=thresh, perfect=False,
                                              model_name=model_name)

        rated_data = {
            'evt_rates': [],  # mean probs
            'soz_labels': []
        }
        elec_count = 0
        for patient in filtered_pat_dic.values():
            elec_count += len(patient.electrodes)
            for e in patient.electrodes:
                filtered_probs = [h.info['proba'][model_name] for h in
                                  e.events[hfo_type_name]]
                mean_prob = np.mean(filtered_probs) if len(
                    filtered_probs) > 0 else 0
                rated_data['evt_rates'].append(mean_prob)
                rated_data['soz_labels'].append(e.soz)

        rated_data['elec_count'] = elec_count
        rated_data['AUC_ROC'] = roc_auc_score(rated_data['soz_labels'],
                                              rated_data['evt_rates'])
        event_type_data_by_loc[loc_name][
            hfo_type_name + model_name + ' mean prob ' + ' FPR {0}'.format(
                tol_fpr)] = rated_data

    graphics.event_rate_by_loc(event_type_data_by_loc, metrics=['auc'],
                               title='Hippocampal RonS HFO rate (events per minute) baseline \nVS ML rate, x, y, z, properties.',
                               colors=None)

def Hippocampal_RonS_ml_with_rate(elec_collection, evt_collection, ):
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ml_with_rate(elec_collection, evt_collection, 'Hippocampus', 'RonS',
                 tol_fprs=tol_fprs)



# The two main functions that compare baseline with ml
def compare_event_type_rates_by_loc(elec_collection, evt_collection,
                                    intraop=False, loc_granularity=0,
                                    locations='all',
                                    event_type_names=EVENT_TYPES,
                                    filter_phfos=False, filter_info=None,
                                    bs_info_by_loc=None,
                                    saving_path=EXPERIMENTS_FOLDER):
    print('Comparing event type rates by location...')
    if filter_phfos:
        assert (all(
            [isinstance(filter_info, dict), 'target' in filter_info.keys(),
             'tol_fpr' in filter_info.keys()]))

    patients_by_loc = load_patients(elec_collection, evt_collection,
                                    intraop,
                                    loc_granularity, locations,
                                    event_type_names, models_to_run)

    event_type_data_by_loc = dict()
    print('Populating rate data by location and type...')
    for loc_name, patients_dic in patients_by_loc.items():
        event_type_data_by_loc[loc_name] = dict()
        for evt_type_name in event_type_names:
            event_type_data_by_loc[loc_name][evt_type_name] = region_info(
                patients_dic, [evt_type_name])
            if bs_info_by_loc is not None:
                bs_info_by_loc[loc_name][evt_type_name] = \
                    event_type_data_by_loc[loc_name][evt_type_name][
                        'AUC_ROC']
                bs_info_by_loc[loc_name]['PSE'] = \
                    event_type_data_by_loc[loc_name][evt_type_name][
                        'pse']  # Should agree among types, checked in db_parsing

            if filter_phfos and evt_type_name not in ['Spikes',
                                                      'Sharp Spikes']:
                patients_dic = phfo_filter(evt_type_name, patients_dic,
                                           target=filter_info['target'],
                                           tolerated_fpr=filter_info[
                                               'tol_fpr'],
                                           perfect=filter_info['perfect'])
                event_type_data_by_loc[loc_name][
                    'Filtered ' + evt_type_name] = region_info(patients_dic,
                                                               [
                                                                   evt_type_name],
                                                               flush=True)

    graphics.event_rate_by_loc(event_type_data_by_loc, saving_path)


def compare_subtypes_rate_by_loc(elec_collection, evt_collection,
                                 hfo_type_name,
                                 subtypes='all', loc_granularity=0,
                                 locations='all',
                                 intraop=False, filter_phfos=False,
                                 filter_info=None,
                                 saving_path=EXPERIMENTS_FOLDER):
    # In this case we need to load subtype and calculate the rate to avoid mixing the subtypes in the rate.
    # The structure doesn't differ among subtypes
    subtypes = all_subtype_names(
        hfo_type_name) if subtypes == 'all' else subtypes
    loc, locations = get_locations(loc_granularity, locations)
    subtype_data_by_loc = dict()
    for loc_name in locations:
        subtype_data_by_loc[loc_name] = dict()
        for subtype_name in subtypes:
            elec_filter, hfo_filter = query_filters(intraop,
                                                    [hfo_type_name],
                                                    loc, loc_name,
                                                    [subtype_name])
            elec_cursor = elec_collection.find(elec_filter,
                                               projection=electrodes_query_fields)
            hfo_cursor = evt_collection.find(hfo_filter,
                                             projection=hfo_query_fields)
            patients_dic = parse_patients(elec_cursor, hfo_cursor,
                                          models_to_run)
            subtype_data_by_loc[loc_name][subtype_name] = region_info(
                patients_dic, [hfo_type_name])

            if filter_phfos:
                patients_dic = phfo_filter(hfo_type_name, patients_dic,
                                           target=filter_info['target'],
                                           tolerated_fpr=filter_info[
                                               'tol_fpr'],
                                           perfect=filter_info['perfect'])
                subtype_data_by_loc[loc_name][
                    'Filtered_' + hfo_type_name] = region_info(patients_dic,
                                                               [
                                                                   hfo_type_name])

    graphics.event_rate_by_loc(subtype_data_by_loc,
                               zoomed_type=hfo_type_name,
                               roc_saving_path=saving_path)
    plt.show()

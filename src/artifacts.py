from random import choices
import numpy as np

# TODO correct with actual ratio of 300/360 and 180/120
# If we use kmeans
# predictors = ['freq_av', 'duration', 'power_pk']
# data = {predictor: [] for predictor in predictors}
# for pred in predictors:
#    data[pred].append(evt.info[pred])
# graphics.k_means_clusters_plot(data)
from random import choices


def artifact_filter(hfo_type, patients_dic):
    '''
    Filters Fast RonO near 300 HZ and RonO near 180 Hz electrical artifacts.
    :param hfo_type: The hfo type with electrical artifacts
    :param patients_dic: Patient, Electrode, Event data structures
    :return: modified patients dic for type hfo_type
    '''
    print('Entering filter for electrical artifacts')
    remove_from_elec_by_pat = {p_name: [] for p_name in patients_dic.keys()}
    # For each patient I keep a list of elec names where we can gradually
    # remove candidates and update
    if hfo_type == 'Fast RonO':
        artifact_freq = 300  # HZ
        art_radius = 20  # HZ
        pw_line_int = 60  # HZ
        artifact_cnts = dict()  # 300 HZ +- art_radius event counts for each patient
        physio_cnts = []  # 360 HZ +- art_radius event counts for each patient
        for p_name, p in patients_dic.items():
            artifact_cnt = 0
            physio_cnt = 0
            for e in p.electrodes:
                for evt in e.events['Fast RonO']:
                    if (artifact_freq - art_radius) <= evt.info['freq_av'] and \
                            evt.info['freq_av'] <= (artifact_freq + \
                                                    art_radius):
                        artifact_cnt += 1
                        remove_from_elec_by_pat[p_name].append(e.name)

                    elif (artifact_freq + pw_line_int - art_radius) <= evt.info[
                        'freq_av'] and \
                            evt.info['freq_av'] <= (artifact_freq +
                                                    pw_line_int + art_radius):
                        physio_cnt += 1
            artifact_cnts[p_name] = artifact_cnt
            physio_cnts.append(physio_cnt)

        # Saving stats
        artifact_mean = np.mean(list(artifact_cnts.values()))
        # artifact_mean = np.median(list(artifact_cnts.values()))
        artifact_std = np.std(list(artifact_cnts.values()), ddof=1)
        physio_mean = np.mean(physio_cnts)
        # physio_mean = np.median(list(physio_cnts))
        physio_std = np.std(physio_cnts, ddof=1)
        print('-----------------------------------')
        print('\nFRonO Artifacts (300 HZ +- {0})'.format(art_radius))
        print('Sample artifact mean', artifact_mean)
        print('Sample artifact std', artifact_std)
        print('\nFRonO Physiological (360 HZ +- {0})'.format(art_radius))
        print('Sample physiological mean', physio_mean)
        print('Sample physiological std', physio_std)
        print('-----------------------------------')
        # Removing artifacts
        for p_name, p in patients_dic.items():
            remove_cnt = max(0, int(artifact_cnts[p_name] - physio_mean))
            print('For patient {0} we remove {1} events'.format(p_name,
                                                                remove_cnt))
            for i in range(remove_cnt):
                elec_to_rmv = choices(remove_from_elec_by_pat[p_name], k=1)[0]
                remove_from_elec_by_pat[p_name].remove(elec_to_rmv)
                electrode = p.get_electrode(elec_to_rmv)
                electrode.remove_rand_evt(hfo_type='Fast RonO',
                                          art_radius=art_radius)

            for e in p.electrodes:
                e.flush_cache(
                    ['Fast RonO'])  # Recalc events counts for hfo rate

    else:
        print('Not implemented filter type')
        raise NotImplementedError()

    return patients_dic

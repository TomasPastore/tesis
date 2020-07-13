import warnings
from sys import version as py_version

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import ranksums, ks_2samp, mannwhitneyu
from config import (HFO_TYPES, exp_save_path)
from db_parsing import get_granularity

running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    pass
import graphics
import math as mt



# 2) HFO rate in SOZ vs NSOZ  ##################################################################
# Note: HFO rate is defined in patient.py module as a method for the electrode object
def hfo_rate_statistical_tests(patients_dic, types=HFO_TYPES,
                               saving_path=exp_save_path[2]['dir']):
    # Getting HFO rate for SOZ and NSOZ for each type
    data_by_type = {type:{'soz': [], 'nsoz': []} for type in types}
    for p in patients_dic.values():
        for e in p.electrodes:
            soz_label = 'soz' if e.soz else 'nsoz'
            for t in types:
                data_by_type[t][soz_label].append(e.get_events_rate([t]))

    # Calculating Stat and pvalue and plotting
    stats = dict( HFO_rate={type: dict() for type in types})
    feat_name = 'HFO_rate'
    for t in types:
        data_soz = data_by_type[t]['soz']
        data_nsoz = data_by_type[t]['nsoz']
        if min(len(data_soz), len(data_nsoz) )== 0:
            print('There is no info for type {t}'.format( t=t))
        else:

            test_names = {'D': 'Kolmogorov-Smirnov test',
                          'W': 'Wilcoxon signed-rank test',
                          'U': 'Mann-Whitney U test'}
            test_func = {'D': ks_2samp,
                         'W': ranksums,
                         'U': mannwhitneyu}
            for s_name, test_f in test_func.items():
                stats[feat_name][t][test_names[s_name]] = dict()
                stats[feat_name][t][test_names[s_name]][s_name], \
                stats[feat_name][t][test_names[s_name]]['pval'] = \
                    test_f(data_soz, data_nsoz)
            graphics.plot_feature_distribution(data_soz,
                                               data_nsoz,
                                               feature=feat_name,
                                               type=t,
                                               stats=stats,
                                               test_names=test_names,
                                               saving_path=saving_path)


# 4) ML HFO classifiers
# Compare modelos con y sin balanceo, el scaler, la forma de hacer la particion de pacientes y param tuning
# Model patients da peor

# Stats
# TODO move results to overleaf
def feature_statistical_tests(patients_dic,
                              location=None,
                              types=HFO_TYPES,
                              features=['duration', 'freq_pk', 'power_pk'],
                              saving_path=exp_save_path[4]['dir']):
    # Structure initialization
    feature_data = dict()
    stats = dict()
    for feature in features:
        if 'angle' in feature:
            feature_data['sin_' + feature] = dict()
            feature_data['cos_' + feature] = dict()
            stats['sin_' + feature] = dict()
            stats['cos_' + feature] = dict()
        else:
            feature_data[feature] = dict()
            stats[feature] = dict()

        for t in types:
            if 'angle' in feature:
                feature_data['sin_' + feature][t] = {'soz': [], 'nsoz': []}
                feature_data['cos_' + feature][t] = {'soz': [], 'nsoz': []}
            else:
                feature_data[feature][t] = {'soz': [], 'nsoz': []}

    granularity = get_granularity(location)
    # Gathering data
    for p in patients_dic.values():
        if location is None:
            electrodes = p.electrodes
        else:
            electrodes = [e for e in p.electrodes if
                          location == getattr(e, 'loc{g}'.format(g=
                          granularity))]
        for e in electrodes:
            soz_label = 'soz' if e.soz else 'nsoz'
            for t in types:
                for h in e.events[t]:
                    for f in features:
                        if 'angle' in f:
                            if h.info[f[:-len('_angle')]] == True:
                                feature_data['sin_{f}'.format(f=f)][t][
                                    soz_label].append(mt.sin(h.info[f]))
                                feature_data['cos_{f}'.format(f=f)][
                                t][soz_label].append(mt.cos(h.info[f]))
                        else:
                            feature_data[f][t][soz_label].append(h.info[f])

    # Calculating Stat and pvalue and plotting
    for f in features:
        if 'angle' in f:
            f_names = ['sin_{f}'.format(f=f), 'cos_{f}'.format(f=f)]
        else:
            f_names = [f]
        for t in types:
            for feat_name in f_names:
                if min(len(feature_data[feat_name][t]['soz']),
                       len(feature_data[feat_name][t]['nsoz'])) == 0:
                    print('There is no info for {f} with type {t}'.format(f=feat_name, t=t))
                else:
                    stats[feat_name][t] = dict()
                    data_soz = feature_data[feat_name][t]['soz']
                    data_nsoz = feature_data[feat_name][t]['nsoz']
                    test_names = {'D': 'Kolmogorov-Smirnov test',
                                  'W': 'Wilcoxon signed-rank test',
                                  'U': 'Mann-Whitney U test'}
                    test_func = {'D': ks_2samp,
                                  'W': ranksums,
                                  'U': mannwhitneyu}
                    for s_name, test_f in test_func.items():
                        stats[feat_name][t][test_names[s_name]] = dict()
                        stats[feat_name][t][test_names[s_name]][s_name], \
                        stats[feat_name][t][test_names[s_name]]['pval']= \
                            test_f(data_soz, data_nsoz)
                    graphics.plot_feature_distribution(data_soz,
                                                       data_nsoz,
                                                       feature=feat_name,
                                                       type=t,
                                                       stats=stats,
                                                       test_names=test_names,
                                                       saving_path=saving_path)


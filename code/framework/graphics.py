from decimal import Decimal

import numpy as np
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

from config import EVENT_TYPES
from utils import angle_clusters


def encode_type_name(name):
    return str(EVENT_TYPES.index(name) + 1)


# Graphics for analysis
def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name


# Esta funcion es la que generó los gráficos del primer paper de phase coupling
def compare_hfo_angle_distribution(collection, hfo_type_name, angle_type, fig_title='', angle_step=(np.pi / 9)):
    fig = plt.figure()
    fig.suptitle(fig_title)

    my_dic = {}
    all = collection.find(
        filter={'type': encode_type_name(hfo_type_name), angle_type: 1, 'intraop': '0', 'loc5': 'Hippocampus'},
        projection=['patient_id', 'electrode'])
    for a in all:
        if a['patient_id'] not in my_dic.keys():
            my_dic[a['patient_id']] = set()

        my_dic[a['patient_id']].add(parse_elec_name(a))
    elec_count = sum([len(elec_set) for elec_set in my_dic.values()])
    print('group by manual: {0} '.format(elec_count))

    for soz_str in ['0', '1']:

        hfo_filter = {'type': encode_type_name(hfo_type_name),
                      angle_type: 1,
                      'intraop': '0',
                      'soz': soz_str,
                      'loc5': 'Hippocampus'}

        count_by_group, mean_angle, pvalue, hfo_count, elec_count = angle_clusters(collection=collection,
                                                                                   hfo_filter=hfo_filter,
                                                                                   angle_field_name=angle_type + '_angle',
                                                                                   amp_step=angle_step,
                                                                                   )
        z_value = -np.log(pvalue)  # vale porque angles.size >= 50, ver el codigo fuente de rayleightest

        angles = []
        values = []
        for k, v in count_by_group.items():
            angles.append(angle_step * float(k) + angle_step / 2)
            values.append(v)

        axe = plt.subplot(int('12' + str(int(soz_str) + 1)), polar=True)
        title = 'SOZ' if soz_str == '1' else 'NSOZ'
        # elec_count = 77 if soz_str == '1' else 80

        axe.set_title(title, fontdict={'fontsize': 16}, pad=10)
        polar_bar_plot(fig, axe, angles, values, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count,
                       z_value=z_value, elec_count=elec_count)

    plt.show()


def polar_bar_plot(fig, ax1, angles, values, mean_angle, pvalue, hfo_count, z_value=0, elec_count=0):
    heights = values
    bars = ax1.bar(angles,
                   heights,
                   align='center',
                   # color='xkcd:salmon',
                   color=plt.cm.magma(heights),
                   width=0.1,
                   bottom=0.0,
                   edgecolor='k',
                   alpha=0.5,
                   label='HFO count (%)')
    annot = ax1.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                         bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                         arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    # max_count = max(values)
    max_value = max(values)
    radius_limit = max_value + (10 - max_value % 10)  # finds next 10 multiple
    ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    ax1.set_ylim(0, max_value)
    ax1.set_yticks(np.linspace(0, radius_limit, 5))
    ax1.grid(True)
    ax1.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.05, 1.05))
    info_txt = 'Electrode count: {count}'.format(count=elec_count)
    ax1.text(-0.15, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)
    raleigh_txt = ('Rayleigh Test \n \n'
                   'Z-value: {zvalue} \n'
                   'P-value: {pvalue} \n'
                   'Mean: {mean}° \n').format(zvalue="{:.2E}".format(Decimal(z_value)),
                                              pvalue="{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))
    ax1.text(-0.15, 0, raleigh_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    def update_annot(bar):
        x = bar.get_x() + bar.get_width() / 2.
        y = bar.get_y() + bar.get_height()
        annot.xy = (x, y)
        text = "{c}".format(c=y)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax1:
            for bar in bars:
                cont, ind = bar.contains(event)
                if cont:
                    update_annot(bar)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


# Graphics for results

def event_rate_by_loc(hfo_type_data_by_loc, zoomed_type=None, legend_metrics=['ec','pewp', 'cs', 'ps']):
    fig = plt.figure(107)

    # Subplots frames
    subplot_count = len(hfo_type_data_by_loc.keys())
    if subplot_count == 1:
        rows = 1
        cols = 1
    elif subplot_count < 5:
        rows = 2
        cols = 2
    elif subplot_count < 7:
        rows = 2
        cols = 3
    elif subplot_count < 10:
        rows = 3
        cols = 3
    else:
        raise RuntimeError('Subplot count not implemented')

    subplot_index = 1
    if zoomed_type is None:
        fig.suptitle('Event types\' rate (events per minute)')
    else:
        fig.suptitle('{0} subtypes\' rate (events per minute)'.format(zoomed_type))

    for loc, rate_data_by_type in hfo_type_data_by_loc.items():
        axe = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        oracles = []
        preds = []
        legends = []
        for type, rate_data in rate_data_by_type.items():
            oracles.append(rate_data['soz_labels'])
            preds.append(rate_data['evt_rates'])
            legend = '{t}.'.format(t=type)
            if type not in ['Spikes', 'Sharp Spikes']:
                for m in legend_metrics:
                    if m == 'ec':
                        legend = legend + ' EC {ec}'.format(ec=rate_data['evt_count'])
                    if m == 'pewp':
                        legend = legend + ' PEWP {pewp}%'.format(pewp=rate_data['p_elec_with_pevts'])
                    if m == 'cs':
                        legend = legend + ' CS {cs}%'.format(cs=rate_data['capture_score'])
                    if m == 'ps':
                        legend = legend + ' PS {prop_score}%'.format(prop_score=rate_data['proportion_score'])
            else:
                for m in legend_metrics:
                    if m == 'ec':
                        legend = legend + ' EC {ec}'.format(ec=rate_data['evt_count'])
                    if m == 'pewp':
                        legend = legend + ' PEWP {pewp}%'.format(pewp=rate_data['p_elec_with_pevts'])
                    if m == 'ps':
                        legend = legend + ' PS {prop_score}%'.format(prop_score=rate_data['proportion_score'])
            legends.append(legend)
            subplot_title = '{l}'.format(l=loc)

        superimposed_rocs(oracles, preds, legends, subplot_title, axe, rate_data['elec_count'])
        subplot_index += 1

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def superimposed_rocs(oracles, preds, legends, title, axe, elec_count):
    axe.set_title(title, fontdict={'fontsize': 10}, loc='left')
    axe.plot([0, 1], [0, 1], 'r--')
    axe.set_xlim([0, 1])
    axe.set_ylim([0, 1])
    axe.set_ylabel('True Positive Rate')
    axe.set_xlabel('False Positive Rate')
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'lightcoral', 'mediumslateblue']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(oracles)):
        fpr, tpr, threshold = metrics.roc_curve(oracles[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        axe.plot(fpr, tpr, colors[i], label=legends[i] + '. AUC = %0.2f' % roc_auc)

    axe.legend(loc='lower right', prop={'size': 8})
    axe.text(0.03, 0.92, 'Elec Count: {0}'.format(elec_count), bbox=dict(facecolor='grey', alpha=0.5),
             transform=axe.transAxes, fontsize=8)

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')


def feature_importances(feature_list, importances, hfo_type_name, fig_id):
    fig = plt.figure(fig_id)
    axe = plt.subplot(111)
    # plt.style.use('fivethirtyeight')

    # Vertical bars
    # x_values = list(range(len(importances)))
    # axe.bar(importances, x_values, orientation='horizontal')
    # plt.xticks(x_values, feature_list ) #rotation='vertical'

    # Horizontal bars
    pos = np.arange(len(feature_list))
    rects = axe.barh(pos, importances,
                     align='center',
                     height=0.5,
                     tick_label=feature_list)
    axe.set_title('{0} Variable Importances'.format(hfo_type_name))
    axe.set_ylabel('Importance')
    axe.set_xlabel('Variable')
    plt.show()


def hfo_rate_histogram(data, title):
    print('{0} Histogram'.format('HFO rate per electrode'))
    fig = plt.figure(19)
    weight_for_obs_i = 1. / len(data)
    weights = [weight_for_obs_i] * len(data)  # Option 1

    datas_not_0 = [x for x in data if x > 0]
    min_rate = min(datas_not_0)
    print('Min rate after 0: {0}'.format(min_rate))
    n, bins, patches = plt.hist(
        data,
        bins=[0., 0.015] + list(np.linspace(0.015, 15, 150)),
        weights=weights,
        histtype='step',
        color='r',
        label='HFO rate'
        # stacked=True,
    )
    plt.legend(loc='upper right')
    print(n)
    x_label = 'HFO rate'
    plt.xlabel(x_label)
    plt.ylabel('Proportion of electrodes')
    plt.title(title)
    plt.show()


def physiological_vs_pathological_feature_distributions(feature_name, soz_data, n_soz_data):
    # Usage example 'Power Peak', [soz hfos pw pk] [n_soz hfos pw pk]
    print('{0} Histogram'.format(feature_name))

    # example data
    # mu = 100 # mean of distribution
    # sigma = 15 # standard deviation of distribution

    data = [soz_data, n_soz_data]
    soz_color = 'r'
    n_soz_color = 'k'

    rang = (3, 10) if feature_name == 'Power Peak' else None

    # Uncomment for option 1 --> soz + n_soz  == 1  Para mi la correcta es la 2
    #weight_for_obs_i = 1. / (len(data[0]) + len(data[1]))
    #weights = [[weight_for_obs_i] * len(data[0]), [weight_for_obs_i] * len(data[1])]  # Option 1

    # Uncomment for option 2 --> soz == 1 n_soz == 1 --> soz + n_soz == 2
    weight_for_obs_i_soz = 1./len(data[0])
    weight_for_obs_i_nsoz = 1./len(data[1])
    weights = [ [weight_for_obs_i_soz]*len(data[0]), [weight_for_obs_i_nsoz]*len(data[1]) ]

    n, bins, patches = plt.hist(
        data,
        weights=weights,
        range=rang,
        histtype='step',
        color=[soz_color, n_soz_color],
        label=['SOZ', 'N_SOZ'],
        # stacked=True,
     )

    print('Sum of bar heights for Soz: {0}'.format(sum(n[0])))
    print('Sum of bars heights for N_Soz: {0}'.format(sum(n[1])))
    print('Total sum of bar heights soz+n_soz: {0}'.format(sum(n[0]) + sum(n[1])))

    plt.legend(loc='lower right')

    # add a 'best fit' line
    # y = mlab.normpdf(bins, mu, sigma)
    # plt.plot(bins, y, 'r--')

    if feature_name == 'Duration':
        x_label = 'Duration (milliseconds)'
    elif feature_name == 'Spectral content':
        x_label = 'Frequency Peak'
    elif feature_name == 'Power Peak':
        x_label = feature_name

    plt.xlabel(x_label)
    plt.ylabel('Probability')
    plt.title('Distribution of Hippocampal RonO {0}'.format(feature_name))

    # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    plt.show()


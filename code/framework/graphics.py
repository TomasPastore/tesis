from decimal import Decimal

import numpy as np
import math as mt
import sklearn.metrics as metrics
from matplotlib import pyplot as plt, colors

from config import EVENT_TYPES, intraop_patients, HFO_TYPES
from utils import angle_clusters, get_matlab_session, get_granularity
import matplotlib.style as mplstyle
# mplstyle.use(['ggplot', 'fast'])

def encode_type_name(name): #TODO sacar
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


# Graphics for results

def event_rate_by_loc(hfo_type_data_by_loc, zoomed_type=None, metrics=['pewp', 'cs', 'ps', 'ms', 'auc']):
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
        plot_data = {type: {} for type in rate_data_by_type.keys()}
        for type, rate_data in rate_data_by_type.items():
            plot_data[type]['labels'] = rate_data['soz_labels']
            plot_data[type]['preds'] = rate_data['evt_rates']
            plot_data[type]['legend'] = '{t}.'.format(t=type)
            scores = {}
            if 'ec' in metrics:
                scores['EC'] = rate_data['evt_count']
            if 'pewp' in metrics:
                scores['PEWP'] = rate_data['p_elec_with_pevts']
            if 'ps' in metrics:
                scores['PS'] = round(rate_data['proportion_score'], 2)
            if 'auc' in metrics:
                scores['AUC_ROC'] = round(rate_data['AUC_ROC'], 2)

            if type not in ['Spikes', 'Sharp Spikes']:
                if 'cs' in metrics:
                    scores['CS'] = round(rate_data['capture_score'], 2)
                if 'ms' in metrics:
                    scores['MS'] = round(rate_data['metric_score'], 2)
            else:
                if 'cs' in metrics:
                    scores['CS'] = 'X'
                if 'ms' in metrics:
                    scores['MS'] = 'X'

            plot_data[type]['scores'] = scores
            title = '{l}'.format(l=loc)
        print('Plot data {0}'.format(plot_data))
        superimposed_rocs(plot_data, title, axe, rate_data['elec_count'])
        subplot_index += 1
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()


def superimposed_rocs(plot_data, title, axe, elec_count):
    axe.set_title(title, fontdict={'fontsize': 10}, loc='left')
    axe.plot([0, 1], [0, 1], 'r--')
    axe.set_xlim([0, 1])
    axe.set_ylim([0, 1])
    axe.set_ylabel('True Positive Rate')
    axe.set_xlabel('False Positive Rate')
    colors = ['b', 'g', 'c', 'm', 'y', 'k', 'lightcoral', 'mediumslateblue']

    # calculate the fpr and tpr for all thresholds of the classification
    roc_data = []
    for type, info in plot_data.items():
        fpr, tpr, threshold = metrics.roc_curve(info['labels'], info['preds'])
        roc_data.append((type, fpr, tpr, info['scores']['AUC_ROC']))

    roc_data.sort(key=lambda x: x[3], reverse=True)
    columns = []
    rows = []
    for t, fpr, tpr, auc in roc_data:
        legend = plot_data[t]['legend'] + ' AUC_ROC %.2f' % auc
        axe.plot(fpr, tpr, color_for(t), label=legend)
        columns = [title] + [k for k in ['EC', 'PEWP', 'CS', 'PS', 'MS', 'AUC_ROC'] if
                             k in plot_data[t]['scores'].keys()]
        rows.append([t] + [str(plot_data[t]['scores'][k]) for k in ['EC', 'PEWP', 'CS', 'PS', 'MS', 'AUC_ROC'] if
                           k in plot_data[t]['scores'].keys()])
    axe.legend(loc='lower right', prop={'size': 7})
    axe.text(0.03, 0.92, 'Elec Count: {0}'.format(elec_count), bbox=dict(facecolor='grey', alpha=0.5),
             transform=axe.transAxes, fontsize=8)

    plot_score_in_loc_table(columns, rows)

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')


def color_for(t):
    if t == 'HFOs':
        return 'b'
    if t == 'RonO':
        return 'b'
    if t == 'RonS':
        return 'g'
    if t == 'Fast RonO':
        return 'm'
    if t == 'Fast RonS':
        return 'y'
    if t == 'Spikes':
        return 'c'
    if t == 'Sharp Spikes':
        return 'k'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'

    print('No deberia llegar aca')
    print(t)

def table_color_for(t):
    if t == 'HFOs':
        return 'blue'
    if t == 'RonO':
        return 'blue'
    if t == 'RonS':
        return 'green'
    if t == 'Fast RonO':
        return 'magenta'
    if t == 'Fast RonS':
        return 'yellow'
    if t == 'Spikes':
        return 'lightcyan'
    if t == 'Sharp Spikes':
        return 'black'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'

    print('No deberia llegar aca, table color for')
    print(t)

def color_by_gran(granularity):
    if granularity == 0:
        return 'lightblue'
    elif granularity == 2:
        return 'lightsalmon'
    elif granularity == 3:
        return 'lightgreen'
    elif granularity == 5:
        return 'lightyellow'
    else:
        raise RuntimeError('Undefined color for granularity {0}'.format(granularity))


def plot_score_in_loc_table(columns, rows):
    import plotly.graph_objects as go
    col_colors = [table_color_for(t) for t in sorted([r[0] for r in rows])]
    font_colors = ['black' if c != 'blue' else 'white' for c in col_colors]
    rows = sorted(rows, key=lambda x: x[0])
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['<b>' + c + '</b>' for c in columns],
                line_color='black', fill_color='white',
                align='left', font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[[r[i] for r in rows] for i in range(len(columns))],
                fill_color=[np.array(col_colors) for i in range(len(columns))],
                align='left', font=dict(color=[font_colors for i in range(len(columns))], size=12)
            ))
        ])
    fig.show()



def plot_score_table(t1, name):
    import plotly.graph_objects as go
    np.random.seed(1)
    col_colors = []
    rows = []
    for loc, type_info in t1.items():
        row = []
        granularity = get_granularity(loc)
        row.append(granularity)
        row.append(loc)
        for type, v in sorted(list(type_info.items()), key=lambda p: p[0]):
            row.append(round(v, 2))
        rows.append(tuple(row))

    rows = sorted(rows, key=lambda x: (x[0], x[1]))
    for row in rows:
        col_colors.append(color_by_gran(row[0]))

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['<b>Granularity</b>', '<b>location</b>', '<b>Fast RonO</b>',
                        '<b>Fast RonS</b>', '<b>RonO</b>', '<b>RonS</b>'],
                line_color='black', fill_color='white',
                align='left', font=dict(color='black', size=10)
            ),
            cells=dict(
                values=[[r[i] for r in rows] for i in range(6)],
                fill_color=[np.array(col_colors) for i in range(6)],
                align='left', font=dict(color='black', size=8)
            ))
        ])

    fig.show()


def plot_co_metric_auc(m_tab, a_tab):
    ms = []
    aucs = []
    for loc in m_tab.keys():
        for t in HFO_TYPES:
            ms.append(m_tab[loc][t])
            aucs.append(a_tab[loc][t])
    fig = plt.figure()
    plt.scatter(ms, aucs)
    axes = plt.gca()
    m, b = np.polyfit(ms, aucs, 1)
    X_plot = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 100)
    plt.plot(X_plot, m * X_plot + b, '-')
    plt.xlabel('Pathologic score')
    plt.ylabel('AUC ROC7')
    plt.title('Metric and AUC ROC correlation')
    plt.savefig("/home/tpastore/metric_auc_correlation.png", bbox_inches='tight')
    plt.show()


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

def histogram(data, title, label, x_label, bins=None):
    plt.figure()
    weight_for_obs_i = 1. / len(data)
    weights = [weight_for_obs_i] * len(data)
    print('Phfo stats: {0}'.format(title))
    print(sorted(data))
    print('Mean: {0}'.format(np.mean(data)))
    print('Std: {0}'.format(np.std(data)))
    print('Median: {0}'.format(np.median(data)))

    n, bins, patches = plt.hist(
        data,
        bins=bins,
        weights=weights,
        histtype='step',
        color='r',
        label=label
        # stacked=True,
    )
    print(bins)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


def barchart(red, green, yellow, orange):
    labels = ['SOZ elec with phfos', 'NSOZ elec without phfos', 'NSOZ elec with phfos', 'SOZ elec without phfos']
    fig = plt.figure()
    ax = plt.subplot(111)
    print('print color counts')
    print('red {0}, green {1}, yellow {2}, orange {3}'.format(red,green,yellow,orange))
    x= [1,2,3,4]
    ax.bar(1, red, color='r') #ok
    ax.bar(2, green, color='g') #ok
    ax.bar(3, yellow, color='y') #ok
    ax.bar(4, orange, color='orange') #ok

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Elec color count')
    plt.show()


# Paper phase coupling

def compare_hfo_angle_distribution(collection, hfo_type_name, angle_type, fig_title='', angle_step=(np.pi / 9)):
    fig = plt.figure()
    fig.suptitle(fig_title)

    # Cuenta cuantos electrodos hay concatenando los de los pacientes no intraoperatorios
    my_dic = {}
    all = collection.find(
        filter={'patient_id': {'$nin': intraop_patients}, 'type': encode_type_name(hfo_type_name), angle_type: 1,
                'intraop': '0', 'loc5': 'Hippocampus'},
        projection=['patient_id', 'electrode'])
    for a in all:
        if a['patient_id'] not in my_dic.keys():
            my_dic[a['patient_id']] = set()

        my_dic[a['patient_id']].add(parse_elec_name(a))
    elec_count = sum([len(elec_set) for elec_set in my_dic.values()])
    print('Elec count of both soz and nsoz: {0} '.format(elec_count))
    ##########

    for soz_str in ['0', '1']:

        hfo_nsoz_filter = {'patient_id': {'$nin': intraop_patients},
                           'type': encode_type_name(hfo_type_name),
                           angle_type: 1,
                           'intraop': '0',
                           'soz': soz_str,
                           'loc5': 'Hippocampus'}

        count_by_group, circ_mean, circ_std, pvalue, k_pval, hfo_count, elec_count = angle_clusters(
            collection=collection,
            hfo_filter=hfo_nsoz_filter,
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

        if hfo_type_name == 'Fast RonO':
            elec_count = 60 if soz_str == '1' else 90
        elif hfo_type_name == 'RonO':
            elec_count = 77 if soz_str == '1' else 135

        axe.set_title(title, fontdict={'fontsize': 16}, pad=10)
        polar_bar_plot(fig, axe, angles, values, circ_mean, circ_std, pvalue, k_pval, hfo_count,
                       z_value=z_value, elec_count=elec_count)

    # plt.savefig("/home/tpastore/{0}.eps".format(fig_title), bbox_inches='tight')
    plt.show()


def polar_bar_plot(fig, ax1, angles, values, circ_mean, circ_std, pvalue, k_pval, hfo_count, z_value=0, elec_count=0):
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

    circ_max = mt.degrees(angles[np.argmax(values)])
    info_txt = ('Electrode count: {count} \n'
                'Circ mean: {mean}° \n'
                'Circ std: {circ_std}°\n'
                'Circ max: {circ_max}°').format(count=elec_count, mean=round(circ_mean), circ_std=round(circ_std),
                                                circ_max=round(circ_max))
    ax1.text(-0.15, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    tests_txt = ('Rayleigh Test \n'
                 'z-value: {zvalue} \n'
                 'p-value: {pvalue} \n'
                 'Kuiper Test \n'
                 'p-value: {k_pval}'
                 ).format(zvalue="{:.2E}".format(Decimal(z_value)),
                          pvalue="{:.2E}".format(Decimal(pvalue)),
                          k_pval=k_pval)
    ax1.text(-0.15, 0, tests_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

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


def hist_feature_distributions(feature_name, soz_data, n_soz_data):
    # Usage example 'Power Peak', [soz hfos pw pk] [n_soz hfos pw pk]
    print('{0} Histogram'.format(feature_name))
    data = [soz_data, n_soz_data]
    soz_color = 'r'
    n_soz_color = 'k'
    rang = (3, 10) if feature_name == 'Power Peak' else None
    # Uncomment for option 1 --> soz + n_soz  == 1  Para mi la correcta es la 2
    # weight_for_obs_i = 1. / (len(data[0]) + len(data[1]))
    # weights = [[weight_for_obs_i] * len(data[0]), [weight_for_obs_i] * len(data[1])]  # Option 1
    # Uncomment for option 2 --> soz == 1 n_soz == 1 --> soz + n_soz == 2
    weight_for_obs_i_soz = 1. / len(data[0])
    weight_for_obs_i_nsoz = 1. / len(data[1])
    weights = [[weight_for_obs_i_soz] * len(data[0]), [weight_for_obs_i_nsoz] * len(data[1])]
    n, bins, patches = plt.hist(
        data,
        bins=30,
        weights=weights,
        range=rang,
        histtype='step',
        color=[soz_color, n_soz_color],
        label=['SOZ', 'N_SOZ'],
        # stacked=True,
    )
    # print('Hist bins {0}'.format(n))
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
        x_label = feature_name + ' (log10)'

    plt.xlabel(x_label)
    plt.ylabel('Probability')
    plt.title('Distribution of Hippocampal RonO {0}'.format(feature_name))

    # Tweak spacing to prevent clipping of ylabel
    # plt.subplots_adjust(left=0.15)
    plt.savefig("/home/tpastore/hist_Hippocampal_RonO_{0}.eps".format(feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/hist_map_Hippocampal_RonO_{0}.png".format(feature_name), bbox_inches='tight')

    plt.show()


def hist2d_feature_distributions(feature_name, data):
    # Usage example 'Power Peak', data of both soz and nsoz
    print('{0} Heat map plot'.format(feature_name))
    if feature_name == 'Duration':
        y_label = 'Duration (milliseconds)'
    elif feature_name == 'Spectral content':
        y_label = 'Frequency Peak'
    elif feature_name == 'Power Peak':
        y_label = feature_name + ' (log10)'
    y_rang = [3, 10] if feature_name == 'Power Peak' else None

    # SOZ
    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d'

    ax.tick_params(grid_color='k', grid_alpha=0.5)

    if feature_name == 'Duration':
        ax.set_yticks(np.arange(0, 180, 20))
        ax.set_ylim((0, 180))
    if feature_name == 'Spectral content':
        ax.set_yticks(np.arange(80, 240, 20))
        ax.set_ylim((80, 240))
    if feature_name == 'Power Peak':
        print('Setting params xticks soz')
        ax.set_yticks(np.arange(4.5, 8.5, 0.5))
        ax.set_ylim((4.5, 8.5))
    x = data['Slow angle']['soz']
    y = data[feature_name]['soz']

    weight_for_obs_i = 1. / len(x)
    weights = [weight_for_obs_i] * len(x)
    hist, xedges, yedges, image = plt.hist2d(x, y, bins=30, cmap=plt.cm.jet)
    print('Sum of bar heights: {0}'.format(sum([sum(row) for row in hist])))
    yrang1 = ax.get_yaxis().get_view_interval()
    print('Rang 1 : {0}'.format(yrang1))
    plt.colorbar()
    plt.xlabel('Slow angle (pi)')
    plt.ylabel(y_label)
    plt.title('Hippocampal RonO {0} by slow angle in SOZ'.format(feature_name))
    if feature_name == 'Duration':
        ax.get_yaxis().set_view_interval(vmin=0, vmax=180, ignore=True)
    if feature_name == 'Spectral content':
        ax.get_yaxis().set_view_interval(vmin=80, vmax=230, ignore=True)
    if feature_name == 'Power Peak':
        ax.get_yaxis().set_view_interval(vmin=4.5, vmax=8, ignore=True)
    yrang1 = ax.get_yaxis().get_view_interval()
    print('Rang 1 : {0}'.format(yrang1))

    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_SOZ.eps".format(feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_SOZ.png".format(feature_name), bbox_inches='tight')

    # NSOZ
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if feature_name == 'Duration':
        ax.set_yticks(np.arange(0, 180, 20))
        ax.set_ylim((0, 180))
    if feature_name == 'Spectral content':
        ax.set_yticks(np.arange(80, 240, 20))
        ax.set_ylim((80, 240))
    if feature_name == 'Power Peak':
        print('Setting params xticks nsoz')
        ax.set_yticks(np.arange(4.5, 8.5, 0.5))
        ax.set_ylim((4.5, 8.5))
    x = data['Slow angle']['n_soz']
    y = data[feature_name]['n_soz']
    weight_for_obs_i = 1. / len(x)
    weights = [weight_for_obs_i] * len(x)
    hist, xedges, yedges, image = plt.hist2d(x, y, bins=30, cmap=plt.cm.jet)
    print('Sum of bar heights: {0}'.format(sum([sum(row) for row in hist])))
    plt.colorbar()
    plt.xlabel('Slow angle (pi)')
    plt.ylabel(y_label)
    if feature_name == 'Duration':
        ax.get_yaxis().set_view_interval(vmin=0, vmax=180, ignore=True)
    if feature_name == 'Spectral content':
        ax.get_yaxis().set_view_interval(vmin=80, vmax=230, ignore=True)
    if feature_name == 'Power Peak':
        ax.get_yaxis().set_view_interval(vmin=4.5, vmax=8, ignore=True)
    yrang2 = ax.get_yaxis().get_view_interval()
    print('Rang 2 : {0}'.format(yrang2))
    plt.title('Hippocampal RonO {0} by slow angle in NSOZ'.format(feature_name))
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_NSOZ.eps".format(feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_NSOZ.png".format(feature_name), bbox_inches='tight')

    # plt.subplots_adjust(left=0.15)
    # plt.show()


def boxplot_feature_distributions(feature_name, data):
    print('{0} Box plot'.format(feature_name))
    if feature_name == 'Duration':
        y_label = 'Duration (milliseconds)'
    elif feature_name == 'Spectral content':
        y_label = 'Frequency Peak'
    elif feature_name == 'Power Peak':
        y_label = feature_name + ' (log10)'

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d'
    soz_data = data[feature_name]['soz']
    nsoz_data = data[feature_name]['n_soz']

    ax.boxplot([soz_data, nsoz_data], labels=['SOZ', 'Non-SOZ'])
    plt.ylabel(y_label)
    plt.title('Hippocampal RonO {0} distribution'.format(feature_name))
    plt.savefig("/home/tpastore/Box_plot_Hippocampal_RonO_{0}.eps".format(feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/Box_plot_Hippocampal_RonO_{0}.png".format(feature_name), bbox_inches='tight')
    plt.show()

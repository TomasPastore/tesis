from decimal import Decimal

import numpy as np
import sklearn.metrics as metrics
from matplotlib import pyplot as plt

from config import HFO_TYPES
from utils import angle_clusters

def encode_type_name(name):
    return str(HFO_TYPES.index(name) + 1)


#Graphics for analysis

def compare_hfo_angle_distribution(collection, hfo_type_name, angle_type, fig_title='', angle_step=(np.pi / 9)):
    fig = plt.figure()
    fig.suptitle(fig_title)
    for soz_str in ['0', '1']:
        hfo_filter = { 'type': encode_type_name(hfo_type_name), angle_type: 1, 'intraop': '0', 'soz':soz_str}
        count_by_group, mean_angle, pvalue, hfo_count = angle_clusters(collection=collection,
                                                                       hfo_filter= hfo_filter,
                                                                       angle_field_name=angle_type+'_angle',
                                                                       amp_step=angle_step,
                                                                       )
        angles = []
        values = []
        for k, v in count_by_group.items():
            angles.append(angle_step * float(k) + angle_step / 2)
            values.append(v)

        axe = plt.subplot(int('12' + str(int(soz_str) + 1)), polar=True)
        title = 'SOZ' if soz_str == '1' else 'NSOZ'
        axe.set_title(title, fontdict={'fontsize': 16}, pad=10)
        polar_bar_plot(fig, axe, angles, values, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count)

    plt.show()


def polar_bar_plot(fig, ax1, angles, values, mean_angle, pvalue, hfo_count):
    # Data
    theta = angles
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

    ## Main tweaks
    # max_count = max(values)
    max_value = max(values)
    radius_limit = max_value + (10 - max_value % 10)  # finds next 10 multiple

    # Angle ticks
    ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    # Radius limits
    ax1.set_ylim(0, max_value)
    # Radius ticks
    ax1.set_yticks(np.linspace(0, radius_limit, 5))

    # Radius tick position in degrees
    # ax1.set_rlabel_position(135)

    # Additional Tweaks
    ax1.grid(True)
    ax1.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.05, 1.05))

    info_txt = 'Total HFO count: {count}'.format(count=hfo_count)
    ax1.text(-0.15, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    raleigh_txt = ('Rayleigh Test \n \n'
                   'P-value: {pvalue} \n'
                   'Mean: {mean}° \n').format(pvalue="{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))
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

def hfo_rate_by_loc(hfo_type_data_by_loc, zoomed_type=None):
    fig = plt.figure()

    #Subplots frames
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
        fig.suptitle('HFO types rate by location (events per minute)')
    else:
        fig.suptitle('{0} subtypes rate by location (events per minute)'.format(zoomed_type))

    for loc, rate_data_by_type in hfo_type_data_by_loc.items():
        axe = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        oracles = []
        preds = []
        legends = []
        for type, rate_data in rate_data_by_type.items():
            oracles.append(rate_data['soz_labels'])
            preds.append(rate_data['hfo_rates'])
            legends.append('{t}. PEWH {pewh}%. HC {hc}'.format(t=type, pewh=rate_data['p_elec_with_hfo'], hc=rate_data['hfo_count']))
            subplot_title = '{l}'.format(l=loc)

        superimposed_rocs(oracles, preds, legends, subplot_title, axe, rate_data['elec_count'])
        subplot_index += 1

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def roc(labels, preds, legend, title='R0C curve'):
    plt.title(title)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    fpr, tpr, threshold = metrics.roc_curve(labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, 'b', label=legend + ' AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.show()


def superimposed_rocs(oracles, preds, legends, title, axe, elec_count):
    axe.set_title(title, fontdict={'fontsize': 10}, loc='left')
    axe.plot([0, 1], [0, 1], 'r--')
    axe.set_xlim([0, 1])
    axe.set_ylim([0, 1])
    axe.set_ylabel('True Positive Rate')
    axe.set_xlabel('False Positive Rate')
    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(oracles)):
        fpr, tpr, threshold = metrics.roc_curve(oracles[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        axe.plot(fpr, tpr, colors[i], label=legends[i] + '. AUC = %0.2f' % roc_auc)

    axe.legend(loc='lower right', prop={'size': 8})
    axe.text(0.03, 0.92, 'EC: {0}'.format(elec_count), bbox=dict(facecolor='grey', alpha=0.5), transform=axe.transAxes,
             fontsize=8)

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')


def feature_importances(feature_list, importances, hfo_type_name):
    fig = plt.figure()
    axe = plt.subplot(111)
    #plt.style.use('fivethirtyeight')

    #Vertical bars
    # x_values = list(range(len(importances)))
    # axe.bar(importances, x_values, orientation='horizontal')
    # plt.xticks(x_values, feature_list ) #rotation='vertical'

    #Horizontal bars
    pos=np.arange(len(feature_list))
    rects = axe.barh(pos, importances,
                     align='center',
                     height=0.5,
                     tick_label=feature_list)
    axe.set_title('{0} Variable Importances'.format(hfo_type_name))
    axe.set_ylabel('Importance')
    axe.set_xlabel('Variable')


def plot_histogram(feature_name, soz_data, n_soz_data):
    # Usage example 'Power Peak', [soz hfos pw pk] [n_soz hfos pw pk]
    print('{0} Histogram'.format(feature_name))

    # example data
    # mu = 100 # mean of distribution
    # sigma = 15 # standard deviation of distribution

    data = [soz_data, n_soz_data]
    soz_color = 'r'
    n_soz_color = 'k'

    rang = (3, 10) if feature_name == 'Power Peak' else None

    # Uncomment for option 1 --> soz + n_soz  == 1
    weight_for_obs_i = 1. / (len(data[0]) + len(data[1]))
    weights = [[weight_for_obs_i] * len(data[0]), [weight_for_obs_i] * len(data[1])]  # Option 1

    # Uncomment for option 2 --> soz == 1 n_soz == 1 --> soz + n_soz == 2
    # weight_for_obs_i_soz = 1./len(data[0])
    # weight_for_obs_i_nsoz = 1./len(data[1])
    # weights = [ [weight_for_obs_i_soz]*len(data[0]), [weight_for_obs_i_nsoz]*len(data[1]) ]

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
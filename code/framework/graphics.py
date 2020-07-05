import copy
from decimal import Decimal
import getpass
from pathlib import Path
from sys import version as py_version
import requests
import numpy as np
import math as mt
import sklearn.metrics as metrics
from matplotlib import pyplot as plt, colors
import plotly.graph_objects as go
import plotly
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score, plot_precision_recall_curve

from config import EVENT_TYPES, intraop_patients, HFO_TYPES, color_list, \
    models_to_run, models_dic, EXPERIMENTS_FOLDER, orca_executable
from models import estimators
running_py_3_5 = py_version[2] == '5'
if running_py_3_5:
    import angle_clusters, get_matlab_session
from db_parsing import get_granularity
import matplotlib.style as mplstyle
# mplstyle.use(['ggplot', 'fast'])

#TODO MOVE TO DB PARSING

def encode_type_name(name):
    return str(EVENT_TYPES.index(name) + 1)

#TODO MOVE TO DB PARSING
# Graphics for analysis
def parse_elec_name(doc):
    if isinstance(doc['electrode'], list):
        e_name = doc['electrode'][0] if len(doc['electrode']) > 0 else None
    elif isinstance(doc['electrode'], str):
        e_name = doc['electrode'] if len(doc['electrode']) > 0 else None
    else:
        raise RuntimeError('Unknown type for electrode name')
    return e_name


# 3) Predicting SOZ with rate

# Plots ROCs for SOZ predictor by hfo rate for different locations and event types
def event_rate_by_loc(hfo_type_data_by_loc, zoomed_type=None, metrics=['pse', 'pnee', 'auc'],
                      title=None, colors=None, conf=None,
                      saving_path=EXPERIMENTS_FOLDER+'fig'):

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
        title = 'Event types\' rate (events per minute)' if title is None else title
        fig.suptitle(title, size=14)
    else:
        fig.suptitle('{0} subtypes\' rate (events per minute)'.format(zoomed_type))

    for loc, rate_data_by_type in hfo_type_data_by_loc.items():
        elec_count = None
        axe = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        title = '{l}'.format(l=loc)
        plot_data = {type: {} for type in rate_data_by_type.keys()}
        for type, rate_data in rate_data_by_type.items():
            plot_data[type]['preds'] = rate_data['evt_rates']
            plot_data[type]['labels'] = rate_data['soz_labels']
            plot_data[type]['legend'] = '{t}.'.format(t=type)
            scores = {}
            if 'ec' in metrics:
                scores['ec'] = rate_data['evt_count']
            if 'pse' in metrics:
                scores['pse'] = rate_data['pse']
            if 'pnee' in metrics:
                scores['pnee'] = rate_data['pnee']
            if 'auc' in metrics:
                scores['AUC_ROC'] = round(rate_data['AUC_ROC'], 2)
            if 'Simulated' in type:
                scores['conf'] = rate_data['conf']
            plot_data[type]['scores'] = scores

            if elec_count is None:
                elec_count = rate_data['elec_count']
            elif elec_count != rate_data['elec_count']:
                print('Elec count of type {0}: {1}, other elec_count {2}'.format(
                    type, rate_data['elec_count'], elec_count) )
                raise RuntimeError('Elec count disagreement among types in ROCs plot')

        superimposed_rocs(plot_data, title, axe, elec_count, colors, saving_path)
        subplot_index += 1
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for fmt in ['pdf', 'png', 'eps']:
        saving_path_f = '{file_path}.{format}'.format(file_path=saving_path, format=fmt)
        plt.savefig(saving_path_f, bbox_inches='tight')
    plt.show()

# Plots the ROCs of many types in a location given in plot_data, modifies the axe object
# It also may build tables of the global info in that location if you uncomment that piece of code
def superimposed_rocs(plot_data, title, axe, elec_count, colors=None,
                      saving_path=None):
    axe.set_title(title, fontdict={'fontsize': 12}, loc='left')
    axe.plot([0, 1], [0, 1], 'r--')
    axe.set_xlim([0, 1])
    axe.set_ylim([0, 1])
    axe.set_ylabel('True Positive Rate', fontdict={'fontsize': 12})
    axe.set_xlabel('False Positive Rate', fontdict={'fontsize': 12})
    # calculate the fpr and tpr for all thresholds of the classification
    roc_data, pses = [], []
    for type, info in plot_data.items():
        fpr, tpr, threshold = metrics.roc_curve(info['labels'], info['preds'])
        confidence = info['scores']['conf'] if 'Simulated' in type else None
        if 'pse' in info['scores'].keys():
            pses.append(info['scores']['pse'])
        roc_data.append((type, fpr, tpr, info['scores']['AUC_ROC'], confidence))

    roc_data.sort(key=lambda x: x[3], reverse=True) #Orders descendent by AUC
    if len(pses) > 0:
        pse = pses[0]
        for e in pses:
            if e != pse:
                #This shouldnt happen, soz electrodes should be independent of the type of them
                raise RuntimeError('SOZ electrode percentage disagreement among evt types of the same location')

    columns, rows = [], []
    i = 0

    #For each type
    for t, fpr, tpr, auc, conf in roc_data:
        legend = plot_data[t]['legend'] + ' AUC_ROC %.2f.' % auc
        axe.plot(fpr, tpr, color_for(t) if colors is None else color_list[i], label=legend)

        #Building report tables
        rows.append([t] + [str(plot_data[t]['scores'][k]) for k in ['ec', 'pse', 'pnee', 'AUC_ROC'] if
                           k in plot_data[t]['scores'].keys()])
        i+=1

    axe.legend(loc='lower right', prop={'size': 10}) #TODO  HACER MAS GRANDE
    info_text = 'Elec Count: {0}'.format(elec_count)
    plot_pse_text = True
    if len(pses) > 0 and plot_pse_text:
        info_text = info_text + '\nPSE:            {0}'.format(round(np.mean(pses), 2))
    axe.text(0.03, 0.88, info_text, bbox=dict(facecolor='grey', alpha=0.5),
             transform=axe.transAxes, fontsize=10)

    columns = [title] + [k for k in ['ec', 'pse', 'pnee', 'AUC_ROC'] if
                         k in plot_data[t]['scores'].keys()] #Title here is the location

    plot_score_in_loc_table(columns, rows, colors, saving_path)

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

# Unique location info table
def plot_score_in_loc_table(columns, rows, colors, saving_path):
    col_colors = [table_color_for(t) if colors is None else color_list[i] for i, t in enumerate(sorted([r[0] for r in rows]))]
    font_colors = ['black' if c != 'blue' else 'white' for c in col_colors]
    rows = sorted(rows, key=lambda x: x[0]) #Order by HFO type name
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

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ))

    if Path(orca_executable).exists():
        print('Orca executable_path: {0}'.format(
            plotly.io.orca.config.executable))
        #plotly.io.orca.config.executable = orca_executable
        #plotly.io.orca.config.save()
        try:
            fig.write_image(saving_path+'_in_loc_table.png')
        except ValueError:
            print('Orca executable is probably invalid, save figure manually.')
    else:
        print('You need to install orca and define orca executable path in '
              'order to plotly tables.')
    fig.show()

# Multiple location info table
def plot_score_table(t1):
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
            columnwidth=[200, 300, 200, 200, 200, 200],
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
    fig.update_layout(width=1300)
    fig.show()



#RELATION TODO
def plot_co_metric_auc_0(m_tab, a_tab):
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
    plt.title('Pathologic score and AUC ROC correlation')
    plt.savefig("/home/tpastore/pscore_auc_correlation.png", bbox_inches='tight')
    plt.show()

def plot_co_metrics_auc(prop_tab, pewp_tab, auc_tab):

    grid = {}
    for t in HFO_TYPES:
        proportions = []
        pewp = []
        aucs = []
        for loc in prop_tab.keys():
            proportions.append(prop_tab[loc][t] if prop_tab[loc][t] < 1 else 0.99 )
            pewp.append(pewp_tab[loc][t] if pewp_tab[loc][t] < 1 else 0.99)
            aucs.append(auc_tab[loc][t] if auc_tab[loc][t] < 1 else 0.99)

        min_prop, max_prop = min(proportions), max(proportions)

        type_grid = {round(i,2) : {round(j,2):[] for j in np.arange(0, 0.5, 0.05)}
                     for i in np.arange(0 ,0.6, 0.1)}
        print('Init type gride {0}'.format(type_grid))
        for i in range(len(proportions)):
            p, pw, auc = proportions[i], pewp[i], aucs[i]
            pw_box =  round(  ( (pw*100) - ((pw*100) % 5) ) /100, 2)
            p_box = round( (p*100 - ((p*100) % 10)) / 100, 2)
            type_grid[p_box][pw_box].append(auc)
        print('GRID BEFORE AVERAGE')
        print(type_grid)
        for f, f_cols in type_grid.items():
            for c, aucs in f_cols.items():
                type_grid[f][c] = np.mean(aucs) if len(aucs)>0 else 0
        print('GRID AFTER AVERAGE')
        print(type_grid)
        grid[t]= type_grid
    heat_map = [[0 for i in range(10)] for j in range(6)]
    for h in HFO_TYPES:
        for f, fcols in grid[h].items():
            for col, auc in fcols.items():
                heat_map[int(f / 0.1)][int(col / 0.05)]= auc

        fig = go.Figure(data=go.Heatmap(
            z=heat_map))
        fig.show()
        '''
        fig = plt.figure()
        fig.suptitle('{0} pHFO and HFO rate AUC relation'.format(h))
        axes = fig.gca()
        axes.set_xlabel('PEWP (proportion of elec with pHFO)')
        axes.set_ylabel('pHFO proportion')
        axes.tick_params(grid_color='k', grid_alpha=0.5)
        weights = []
        x = []
        y = []
        for f, fcols in grid[h].items():
            for col, auc in fcols.items():
                x.append(col + 0.025)
                y.append(f  + 0.05)
                weights.append(auc)
        pw_blocks = np.arange(0, 0.55, 0.05)
        prop_blocks = np.arange(0, 1.1, 0.1)
        hist, xedges, yedges, image = plt.hist2d(x, y, bins=[pw_blocks, prop_blocks], cmap=plt.cm.jet, label='Baseline HFO rate AUC-ROC')
        print('hist, xedges, yedges')
        print(hist)
        print(xedges)
        print(yedges)
        axes.legend(loc='lower right', prop={'size': 7})
        plt.colorbar()
        plt.savefig("/home/tpastore/metrics_auc_correlation.png", bbox_inches='tight')
        plt.show()
        '''



##################     Plotting ML results ROCS and PRE_REC          #######################
def axes_by_model(plt, models_to_run):
    subplot_count = len(models_to_run) * 2
    if subplot_count == 2:
        rows = 2
        cols = 1
    elif subplot_count == 4:
        rows = 2
        cols = 2
    elif subplot_count == 6:
        rows = 2
        cols = 3
    elif subplot_count == 8:
        rows = 2
        cols = 4
    else:
        raise RuntimeError('Subplot count not implemented')
    axes = {}
    subplot_index = 1
    for m in models_to_run:
        if m not in axes.keys():
            axes[m] = {}
        axes[m]['ROC'] = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        subplot_index += 1
    for m in models_to_run:
        axes[m]['PRE_REC'] = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        subplot_index += 1
    return axes


def plot_roc_fold(fpr, tpr, model_name, plot_axe, mean_fpr, tprs, aucs, fold):
    curve_kind = 'ROC'
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    tprs[model_name].append(interp_tpr)
    tprs[model_name][-1][0] = 0.0

    roc_auc = auc(fpr, tpr)
    aucs[model_name][curve_kind].append(roc_auc)
    plot_axe[model_name][curve_kind].plot(fpr, tpr, lw=1, alpha=0.3,
                                          label='Fold %d (AUC = %0.2f)' % (fold, roc_auc))

def average_ROCs(model_name, plot_axe, mean_fpr, tprs, aucs):
    curve_kind = 'ROC'
    plot_axe[model_name][curve_kind].plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs[model_name], axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs[model_name][curve_kind])
    plot_axe[model_name][curve_kind].plot(mean_fpr, mean_tpr, color='b',
                                          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                                          lw=2, alpha=.8)

    std_tpr = np.std(tprs[model_name], axis=0)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    plot_axe[model_name][curve_kind].fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                                  label=r'$\pm$ 1 std. dev.')

    plot_axe[model_name][curve_kind].set_xlim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_ylim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_xlabel('False Positive Rate')
    plot_axe[model_name][curve_kind].set_ylabel('True Positive Rate')
    plot_axe[model_name][curve_kind].set_title('{0} ROC curves'.format(model_name))
    plot_axe[model_name][curve_kind].legend(loc="lower right")


# PRE REC

def plot_pre_rec_fold(recall, precision, model_name, plot_axe, mean_recall, prec, aucs, fold, average_precision):
    curve_kind = 'PRE_REC'
    interp_prec = np.interp(mean_recall, copy.deepcopy(recall), copy.deepcopy(precision))
    prec[model_name].append(interp_prec)
    prec[model_name][-1][0] = 1
    auc_v = auc(recall, precision)
    #precision = [1] + precision
    #recall = [0] + recall

    aucs[model_name][curve_kind].append(auc_v)
    plot_axe[model_name][curve_kind].plot(recall, precision, lw=1, alpha=0.3,
                                          label='Fold %d (AUC = %0.2f. AP = %0.2f.)' % (fold, auc_v, average_precision))


def average_pre_rec(model_name, plot_axe, mean_recall, prec, aucs, aps):
    curve_kind = 'PRE_REC'
    mean_prec = np.mean(prec[model_name], axis=0)
    mean_auc = auc(mean_recall, mean_prec)
    std_auc = np.std(aucs[model_name][curve_kind])
    mean_ap = np.mean(aps[model_name])
    std_ap = np.std(aps[model_name])

    plot_axe[model_name][curve_kind].plot(mean_recall, mean_prec, color='b',
                                          label=r'Mean PRE_REC (AUC = %0.2f $\pm$ %0.2f. AP = %0.2f $\pm$ %0.2f)' % (
                                              mean_auc, std_auc, mean_ap, std_ap),
                                          lw=2, alpha=.8)

    std_prec = np.std(prec[model_name], axis=0)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    plot_axe[model_name][curve_kind].fill_between(mean_recall, prec_lower, prec_upper, color='grey', alpha=.2,
                                                  label=r'$\pm$ 1 std. dev.')

    plot_axe[model_name][curve_kind].set_xlim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_ylim([-0.05, 1.05])
    plot_axe[model_name][curve_kind].set_xlabel('Recall')
    plot_axe[model_name][curve_kind].set_ylabel('Precision')
    plot_axe[model_name][curve_kind].set_title('{0} Precision-Recall curves'.format(model_name))
    plot_axe[model_name][curve_kind].legend(loc="lower right")

##################     Plotting ML results ROCS and PRE_REC          #######################


def plot_pre_rec_fold_2(i, fold, ax, mean_recall, prec, aucs, aps, hfo_type_name, model_name):
    precision, recall, thresholds = precision_recall_curve(fold['test_labels'], fold[model_name]['probs'])
    ap = average_precision_score(fold['test_labels'], fold[model_name]['probs'])
    aps[model_name].append(ap)
    rev_precision = np.array(list(reversed(list(precision))))
    rev_recall = np.array(list(reversed(list(recall))))
    interp_prec = np.interp(mean_recall, copy.deepcopy(rev_recall), copy.deepcopy(rev_precision))
    prec[model_name].append(interp_prec)
    prec[model_name][-1][0] = 1
    auc_v = auc(recall, precision)
    aucs[model_name]['PRE_REC'].append(auc_v)
    label = 'Fold %d (AUC = %0.2f. AP = %0.2f.)' % (i, auc_v, ap)
    model_func = models_dic[model_name]
    clf_preds, clf_probs, clf = model_func(fold['train_features'], fold['train_labels'], fold['test_features'],
                                      feature_list=fold['feature_names'], hfo_type_name=hfo_type_name)
    plot_precision_recall_curve(estimator= clf, X=fold['test_features'], y=fold['test_labels'], name=label, ax=ax )

def ml_training_plot(folds, loc, hfo_type_name, roc=True, pre_rec=True, models_to_run= models_to_run):
    fig = plt.figure()
    fig.suptitle('Phfo models in {0}'.format(loc), fontsize=16)
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    tprs = {m: [] for m in models_to_run}
    prec = {m: [] for m in models_to_run}
    aucs = {m: {'PRE_REC':[], 'ROC':[]} for m in models_to_run}
    aps = {m: [] for m in models_to_run}

    plot_axe = axes_by_model(plt, models_to_run)
    for model_name in models_to_run:
        for i, fold in enumerate(folds):
            # ROC
            fpr, tpr, thresholds = roc_curve(fold['test_labels'], fold[model_name]['probs'])
            plot_roc_fold(fpr, tpr, model_name, plot_axe, mean_fpr, tprs, aucs, i)

            # PRE REC
            if model_name == 'Simulated':
                precision, recall, thresholds = precision_recall_curve(fold['test_labels'], fold[model_name]['probs'])
                precision = np.array(list(reversed(list(precision))))
                recall = np.array(list(reversed(list(recall))))
                #thesholds = np.array(list(reversed(list(thresholds))))
                ap = average_precision_score(fold['test_labels'], fold[model_name]['probs'])
                aps[model_name].append(ap)
                plot_pre_rec_fold(recall, precision, model_name, plot_axe, mean_recall, prec, aucs, i, ap)
            else:
                plot_pre_rec_fold_2(i, fold, plot_axe[model_name]['PRE_REC'], mean_recall, prec, aucs, aps, hfo_type_name, model_name)
        average_ROCs(model_name, plot_axe, mean_fpr, tprs, aucs)
        average_pre_rec(model_name, plot_axe, mean_recall, prec, aucs, aps)

    #plt.savefig('/home/{user}/{type}_phfo_model_comparison_{loc}.png'.format(user=getpass.getuser(), type=hfo_type_name,
    #                                                                         loc=loc), format='png')
    plt.show()




def feature_importances(feature_list, importances, hfo_type_name):
    fig = plt.figure()
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

def hfo_rate_histogram_red_green(red, green, title, bins=None):
    print('{0} Histogram'.format('HFO rate per electrode'))
    fig = plt.figure()
    weight_for_green = 1. / len(green)
    weight_for_red = 1. / len(red)
    weights = [[weight_for_green] * len(green), [weight_for_red] * len(red)]
    axes = fig.gca()
    n, bins, patches = axes.hist(
        [green, red],
        bins=bins,
        weights=weights,
        histtype='step',
        color=['g','r'],
        label=['NSOZ', 'SOZ'],
        align='mid'
        # stacked=True,
    )
    axes.set_xticks(np.arange(0, 80, 5))

    axes.legend(loc='upper right', fancybox=True)
    axes.set_xlabel('HFO rate')
    axes.set_ylabel('Proportion of electrodes')
    axes.set_title(title)
    plt.show()

def histogram(data, title, x_label, bins=None):
    plt.figure()
    weight_for_obs_i = 1. / len(data)
    weights = [weight_for_obs_i] * len(data)
    print('Histogram...')
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
        # stacked=True,
    )
    print(bins)
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

# Auxiliary functions

def color_for(t):
    if 'RonS baseline model_pat' == t:
        return 'blue'
    if 'FPR' in t:
        colors = ['darkred', 'firebrick', 'red', 'indianred', 'lightcoral', 'aquamarine', 'springgreen', 'limegreen', 'green', 'darkgreen']
        return colors[int(t[-1])]
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
    if t == 'Spikes + Sharp Spikes':
        return 'magenta'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'

    raise ValueError('graphics.color_for is undefined for type: {0}'.format(t))

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
    if t == 'Spikes + Sharp Spikes':
        return 'magenta'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'
    raise ValueError('graphics.table_color_for is undefined for type: {0}'.format(t))

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
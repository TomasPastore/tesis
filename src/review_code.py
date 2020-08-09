

'''
TODO review old version
def ml_training_plot_old(target_patients, loc_name, hfo_type, roc=True,
                     pre_rec=False, models_to_run=['XGBoost'],
                     saving_dir):
    from ml_hfo_classifier import gather_folds, print_metrics

    fig = plt.figure()
    fig.suptitle('SOZ HFO classfiers in {0}'.format(loc_name),
                 fontsize=16)

    plot_axe = axes_by_model(plt, models_to_run)
    for model_name in models_to_run:
        # Maps predictions and probas to list in linear search of target pat
        labels, preds, probs = gather_folds(model_name, hfo_type,
                                            target_patients, estimator=np.mean)

        print('Displaying metrics for {t} in {l} ml HFO classifier using {'
              'm}'.format(t=hfo_type, l=loc_name, m=model_name))
        print_metrics(model_name, hfo_type, labels, preds, probs)

        # Plot ROC curve
        curve_kind = 'ROC'
        fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
        plot_axe[model_name][curve_kind].plot(fpr, tpr, lw=1, alpha=0.8,
                                  label='AUC = %0.2f' % metrics.auc(fpr, tpr))
        plot_axe[model_name][curve_kind].plot([0, 1], [0, 1], linestyle='--',
                                         lw=2, color='r', label='Chance',
                                         alpha=.8)
        set_titles('False Positive Rate', 'True Postive Rate', model_name,
                   plot_axe[model_name][curve_kind])

        # PRE REC
        precision, recall, thresholds = metrics.precision_recall_curve(labels,
                                                               probs)
        precision = np.array(list(reversed(list(precision))))
        recall = np.array(list(reversed(list(recall))))
        #thesholds = np.array(list(reversed(list(thresholds))))
        ap = metrics.average_precision_score(labels, probs)
        auc_val = metrics.auc(recall, precision)
        curve_kind= 'PRE_REC'
        plot_axe[model_name][curve_kind].plot(recall, precision, color='b',
                                              label='AUC = %0.2f . AP = %0.2f' % (
                                                  auc_val, ap),
                                              lw=2, alpha=.8)
        set_titles('Recall', 'Precision', model_name,
                   plot_axe[model_name][curve_kind])

    # Saving the figure
    saving_path = str(Path(saving_dir, loc_name, hfo_type, 'ml_train_plot'))
    for fmt in ['pdf', 'png']:
        saving_path_f = '{file_path}.{format}'.format(file_path=saving_dir,
                                                      format=fmt)
        if fmt == 'pdf':
            print('ROC saving path: {0}'.format(saving_path_f))
        plt.savefig(saving_path_f, bbox_inches='tight')
    plt.show()
    plt.close(fig)
'''

# PAPER PHASE COUPLING AUXS
# Need python 3.5 for importing matlab and install matlab_engine package
# TODO review
'''
import sys
import copy
import math as mt
from astropy import units as u
from astropy.stats import rayleightest, kuiper
from scipy.stats import circmean, circstd
from db_parsing import intraop_patients
running_py_3_5 = sys.version[2] == '5'
if running_py_3_5:
    import matlab.engine
running_py_3_5 = sys.version[2] == '5'
if running_py_3_5:
    def angle_clusters(collection, hfo_filter, angle_field_name, amp_step):
        angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
        docs = collection.find(hfo_filter)
        if hfo_filter['soz'] == '0':
            main_soz = False
            hfo_filter_2 = copy.deepcopy(hfo_filter)
            hfo_filter_2['soz'] = '1'
        else:
            main_soz = True
            hfo_filter_2 = copy.deepcopy(hfo_filter)
            hfo_filter_2['soz'] = '0'
        docs2 = collection.find(hfo_filter_2)
        hfo_count = docs.count()
        angles = []
        angles2 = []
        pat_elec = dict()
        for doc in docs:
            angle = doc[angle_field_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
            angles.append(angle)
            angle_group_id = mt.floor(angle / amp_step)
            angle_grouped[str(angle_group_id)] += 1  # increment count of group

            pat_id = doc['patient_id']
            e_name = parse_elec_name(doc)
            if pat_id not in pat_elec.keys():
                pat_elec[pat_id] = set()
            pat_elec[pat_id].add(e_name)

        for k, v in angle_grouped.items():  # normalizing values
            r_value = round((v / hfo_count) * 100, 2)  # show them as relative percentages
            angle_grouped[k] = r_value

        for doc in docs2:
            angle = doc[angle_field_name] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
            angles2.append(angle)

        elec_count = sum([len(elec_set) for elec_set in pat_elec.values()])
        circ_mean = mt.degrees(circmean(angles))
        circ_std = mt.degrees(circstd(angles))

        pvalue = float(rayleightest(np.array(angles) * u.rad))  # doctest: +FLOAT_CMP
        # D, fpp = kuiper(np.array(angles))  # doctest: +FLOAT_CMP
        matlab_engine = get_matlab_session()
        alpha1 = angles2 if main_soz else angles  # nsoz angles
        alpha2 = angles if main_soz else angles2  # soz angles
        matlab_engine.addpath('/home/tpastore/Documentos/lic_computacion/tesis/code/scratch/matlab_circ_package')
        print(len(alpha1))
        print(len(alpha2))

        alpha1 = matlab.double(alpha1)
        alpha2 = matlab.double(alpha2)
        k_pval, k, K = matlab_engine.circ_kuipertest(alpha2, alpha1, nargout=3)
        print(k_pval)
        print(k)
        print(K)
        # D, fpp = 'TODO', 'TODO'  # doctest: +FLOAT_CMP

        return angle_grouped, circ_mean, circ_std, pvalue, k_pval, hfo_count, elec_count

    def histograms(hfo_collection):
        hfos = hfo_collection.find(filter={'patient_id': {'$nin': intraop_patients},
                                           'type': encode_type_name('RonO'),
                                           'slow': 1,
                                           'intraop': '0',
                                           'loc5': 'Hippocampus'},
                                   projection=['soz', 'slow_angle', 'power_pk', 'freq_pk', 'duration'])

        feature_names = ['Duration', 'Spectral content', 'Power Peak', 'Slow angle']
        data = {f_name: dict(soz=[], n_soz=[]) for f_name in feature_names}

        for h in hfos:
            power_pk = float(mt.log10((h['power_pk'])))
            freq_pk = float(h['freq_pk'])
            duration = float(h['duration']) * 1000  # seconds to milliseconds
            slow_angle = float(h['slow_angle']) / np.pi

            soz = parse_soz(h['soz'])
            soz_key = 'soz' if soz else 'n_soz'

            data['Power Peak'][soz_key].append(power_pk)
            data['Spectral content'][soz_key].append(freq_pk)
            data['Duration'][soz_key].append(duration)
            data['Slow angle'][soz_key].append(slow_angle)

        # NORMAL HIST
        # for feature_name in ['Duration', 'Spectral content', 'Power Peak']:
        #    graphics.hist_feature_distributions(feature_name, data[feature_name]['soz'], data[feature_name]['n_soz'])

        # 2D HIST / HEAT MAP
        graphics.hist2d_feature_distributions('Duration', data)
        graphics.hist2d_feature_distributions('Power Peak', data)
        graphics.hist2d_feature_distributions('Spectral content', data)

        # BOX PLOT
        # graphics.boxplot_feature_distributions('Duration', data)
        # graphics.boxplot_feature_distributions('Power Peak', data)
        # graphics.boxplot_feature_distributions('Spectral content', data)


    def get_matlab_session():
        return matlab.engine.start_matlab()

    def phase_coupling_paper_polar(hfo_collection):
    # Phase coupling paper
    graphics.compare_hfo_angle_distribution(hfo_collection, 'RonO', 'slow',
                                            fig_title='RonO slow angle distribution in Hippocampus')
    graphics.compare_hfo_angle_distribution(hfo_collection, 'Fast RonO', 'slow',
                                            fig_title='Fast RonO slow angle distribution in Hippocampus')
    histograms(hfo_collection)  # 2d, 3d

'''



# Graphics old code to review it

# TODO review old code
# Paper phase coupling
'''
def compare_hfo_angle_distribution(collection, hfo_type_name, angle_type,
                                   fig_title='', angle_step=(np.pi / 9)):
    fig = plt.figure()
    fig.suptitle(fig_title)

    # Cuenta cuantos electrodos hay concatenando los de los pacientes no intraoperatorios
    my_dic = {}
    all = collection.find(
        filter={'patient_id': {'$nin': intraop_patients},
                'type': encode_type_name(hfo_type_name), angle_type: 1,
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
        polar_bar_plot(fig, axe, angles, values, circ_mean, circ_std, pvalue,
                       k_pval, hfo_count,
                       z_value=z_value, elec_count=elec_count)

    # plt.savefig("/home/tpastore/{0}.eps".format(fig_title), bbox_inches='tight')
    plt.show()


# TODO review
def polar_bar_plot(fig, ax1, angles, values, circ_mean, circ_std, pvalue,
                   k_pval, hfo_count, z_value=0, elec_count=0):
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
    annot = ax1.annotate("", xy=(0, 0), xytext=(-20, 20),
                         textcoords="offset points",
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
                'Circ max: {circ_max}°').format(count=elec_count,
                                                mean=round(circ_mean),
                                                circ_std=round(circ_std),
                                                circ_max=round(circ_max))
    ax1.text(-0.15, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5),
             transform=ax1.transAxes)

    tests_txt = ('Rayleigh Test \n'
                 'z-value: {zvalue} \n'
                 'p-value: {pvalue} \n'
                 'Kuiper Test \n'
                 'p-value: {k_pval}'
                 ).format(zvalue="{:.2E}".format(Decimal(z_value)),
                          pvalue="{:.2E}".format(Decimal(pvalue)),
                          k_pval=k_pval)
    ax1.text(-0.15, 0, tests_txt, bbox=dict(facecolor='grey', alpha=0.5),
             transform=ax1.transAxes)

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


# TODO review
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
    weights = [[weight_for_obs_i_soz] * len(data[0]),
               [weight_for_obs_i_nsoz] * len(data[1])]
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
    print(
        'Total sum of bar heights soz+n_soz: {0}'.format(sum(n[0]) + sum(n[1])))

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
    plt.savefig(
        "/home/tpastore/hist_Hippocampal_RonO_{0}.eps".format(feature_name),
        bbox_inches='tight')
    plt.savefig(
        "/home/tpastore/hist_map_Hippocampal_RonO_{0}.png".format(feature_name),
        bbox_inches='tight')

    plt.show()


# TODO review
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

    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_SOZ.eps".format(
        feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_SOZ.png".format(
        feature_name), bbox_inches='tight')

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
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_NSOZ.eps".format(
        feature_name), bbox_inches='tight')
    plt.savefig("/home/tpastore/Heat_map_Hippocampal_RonO_{0}_NSOZ.png".format(
        feature_name), bbox_inches='tight')

    # plt.subplots_adjust(left=0.15)
    # plt.show()


# TODO review
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

    ax.boxplot([soz_data, nsoz_data], labels=['SOZ', 'NSOZ'])
    plt.ylabel(y_label)
    plt.title('Hippocampal RonO {0} distribution'.format(feature_name))
    plt.savefig(
        "/home/tpastore/Box_plot_Hippocampal_RonO_{0}.eps".format(feature_name),
        bbox_inches='tight')
    plt.savefig(
        "/home/tpastore/Box_plot_Hippocampal_RonO_{0}.png".format(feature_name),
        bbox_inches='tight')
    plt.show()


# TODO review
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
'''



# param tuning for


'''

# REMOVER MODULO, METER
def param_tuning(hfo_type_name, patients_dic):
    print('Analizying models for hfo type: {0} in {1}... '.format(hfo_type_name, 'Hippocampus'))
    patients_dic, _ = patients_with_more_than(0, patients_dic, hfo_type_name)
    model_patients, validation_patients = pull_apart_validation_set(patients_dic, hfo_type_name)
    model_patient_names = [p.id for p in model_patients]  # Obs mantiene el orden de model_patients
    field_names = ml_field_names(hfo_type_name)
    test_partition = get_balanced_partition(model_patients, hfo_type_name, K=4, method='balance_classes')
    column_names = []
    train_data = []
    labels = []
    partition_ranges = []
    i = 0
    for p_names in test_partition:
        test_patients = [patients_dic[name] for name in p_names]
        x, y = get_features_and_labels(test_patients, hfo_type_name, field_names)
        x_pd = pd.DataFrame(x)
        x_values = x_pd.values
        column_names = x_pd.columns
        scaler = RobustScaler()  # Scale features using statistics that are robust to outliers.
        x_values = scaler.fit_transform(x_values)
        analize_balance(y)
        x_values, y = balance_samples(x_values, y)
        train_data = train_data + list(x_values)
        labels = labels + list(y)
        partition_ranges.append((i, i + len(y)))
        i += len(y)

    data = pd.DataFrame(data=train_data, columns=column_names)
    data['soz'] = labels
    target = 'soz'
    predictors = column_names

    folds = [([i for i in range(len(data)) if (i < t_start or i >= t_end)],
              [i for i in range(t_start, t_end)])
             for t_start, t_end in partition_ranges]

    alg = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=6,
        scale_pos_weight=1,
        seed=7)

    param_test1 = {
        'n_estimators': range(100, 200, 1000),
    }
    param_test2 = {
        'max_depth': range(3, 10, 2),
        'min_child_weight': range(1, 6, 2)
    }

    # grid_search(alg, param_test, folds, fit_features=data[predictors].values, to_labels=data[target].values)

    param_configs = set_param_configs(param_test_2)
    config_result = {id: {'preds': [], 'probs': []} for id in param_configs.keys()}

    for id, c in param_configs.items():
        for train_idx, test_idx in folds:
            print('test_indexes {0}'.format(test_idx))
            train_features = data.iloc[train_idx].drop(columns=['soz']).values
            train_labels = data.iloc[train_idx]['soz']
            test_features = data.iloc[test_idx].drop(columns=['soz']).values
            test_labels = data.iloc[test_idx]['soz']
            alg.fit(train_features, train_labels, eval_metric='aucpr')
            test_predictions = alg.predict(test_features)
            test_probs = alg.predict_proba(test_features)[:, 1]
            config_result[id]['preds'] = config_result[id]['preds'] + list(test_predictions)
            config_result[id]['probs'] = config_result[id]['probs'] + list(test_probs)


def grid_search(alg, param_test, folds, fit_features, to_labels):
     gsearch = GridSearchCV(estimator=alg,
                            param_grid=param_test,
                            scoring='recall',
                            n_jobs=6,
                            iid=False,
                            cv=folds)
     gsearch.fit(fit_features, to_labels)

     print('GRID SEARCH RESULTS ')
     print(gsearch.cv_results_)
     print(gsearch.best_estimator_)
     print(gsearch.best_params_)
     print(gsearch.best_score_)

def get_param_configs(param_test):
     for k, r in param_test.items():
         param_test[k] = [i for i in r]

     i = 0
     permutations = [[]]
     for param_values in param_test.values():
         new_permutations = []
         for v in param_values:
             for p in permutations:
                 new_permutations.append(value)

     param_configs = {}
     id = 1
     for p in permutations:
         param_configs[id] = {list(param_test.keys())[i]: p[i] for i in range(len(list(param_test.keys())))}
         id += 1
     return param_configs

def set_param_config(alg, param_config):
     for feature in param_config.keys():
         alg.set_params(feature=param_config[feature])

#No pude hacerla andar bien con el folds especficado, solo hace esas iteraciones e ignora el early stopping
def modelfit(alg, dtrain, predictors, target, useTrainCV=True, cv_folds=4, folds=None, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                          metrics='aucpr', early_stopping_rounds=early_stopping_rounds, folds=folds) #nfold=cv_folds
        print('Best fold has n_estimators: {0}'.format(cvresult.shape[0]))
        alg.set_params(n_estimators=cvresult.shape[0])
        return alg

'''

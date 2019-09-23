import numpy as np
from astropy import units as u
from astropy.stats import rayleightest
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from utils import angle_clusters
from decimal import Decimal


def rose_plot(collection, angle_step=(np.pi / 9)):
    # Usage example
    # rose_plot(collection,angle_step, 'Brodmann area 28')

    loc_name = 'Amygdala'
    angle_type = 'spike'
    criterion = {'$and': [{'type': "5"}, {angle_type: 1}, {'intraop': '0'}, {'soz': '1'}, {'loc5': loc_name}]}
    count_by_group, step, hfo_count, mean_angle, pvalue = angle_clusters(collection=collection,
                                                                         amp_step=angle_step,
                                                                         crit=criterion)
    angles = []
    values = []
    print('{name}. Count by fase group \n'.format(name=loc_name))
    print(count_by_group)
    for k, v in count_by_group.items():
        angles.append(step * float(k))
        values.append(v)

    polar_bar_plot(angles, values, loc_name=loc_name, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count)


def polar_bar_plot(angles, values, loc_name, mean_angle, pvalue, hfo_count):
    # Data
    theta = angles
    heights = values

    # Get an axes handle/object
    fig = plt.figure()
    ax1 = plt.subplot(111, polar=True, )
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
    plt.grid(True)
    plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.35, 1))
    plt.title("Fast RonS SOZ in {location}".format(location=loc_name, ), fontdict={'fontsize': 16}, pad=10)

    raleigh_txt = ('Rayleigh Test \n \n'
                   'P-value: {pvalue} \n'
                   'Mean: {mean}° \n').format(pvalue="{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))

    fig.text(-0.35, 0, raleigh_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    info_txt = 'Total HFO count: {count}'.format(count=hfo_count)
    fig.text(-0.35, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

    # fig.text(.5, .00, 'Rayleigh test p-value {pvalue}'.format(pvalue=pvalue), ha='center')

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
    plt.show()


# Results
ROC_TITLE = 'Receiver Operating Characteristic.\nHippocampus electrodes.'
def plot_roc(labels, preds, legend, title=ROC_TITLE):
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


ROCS_TITLE = 'Receiver Operating Characteristic by HFO type.\nHippocampus electrodes.'
def plot_rocs(labels, preds, legends, title=ROCS_TITLE):
    plt.title(title)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    colors = ['b', 'g', 'c', 'm', 'y', 'k']

    # calculate the fpr and tpr for all thresholds of the classification
    for i in range(len(labels)):
        fpr, tpr, threshold = metrics.roc_curve(labels[i], preds[i])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, colors[i], label=legends[i] + ' AUC = %0.2f' % roc_auc)

    plt.legend(loc='lower right')
    plt.show()

    # method II: ggplot
    # df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
    # ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')

def feature_importances(feature_list, importances):
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
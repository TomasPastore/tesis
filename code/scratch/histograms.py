import math as mt
import numpy as np
from pymongo import MongoClient

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class Connect(object):
    @staticmethod
    def get_connection():
        return MongoClient("mongodb://localhost:27017")

def soz_bool(db_representation_str):
    return ( True if db_representation_str == "1" else False)


def plot_histogram(feature_name, soz_data, n_soz_data):

    print('{0} Histogram'.format(feature_name))

    # example data
    #mu = 100 # mean of distribution
    #sigma = 15 # standard deviation of distribution

    data = [soz_data, n_soz_data]
    soz_color = 'r'
    n_soz_color = 'k'

    rang = (3, 10) if feature_name == 'Power Peak' else None

    #Uncomment for option 1 --> soz + n_soz  == 1
    weight_for_obs_i = 1./(len(data[0])+len(data[1])) 
    weights = [ [weight_for_obs_i]*len(data[0]), [weight_for_obs_i]*len(data[1]) ] #Option 1

    #Uncomment for option 2 --> soz == 1 n_soz == 1 --> soz + n_soz == 2
    #weight_for_obs_i_soz = 1./len(data[0]) 
    #weight_for_obs_i_nsoz = 1./len(data[1])
    #weights = [ [weight_for_obs_i_soz]*len(data[0]), [weight_for_obs_i_nsoz]*len(data[1]) ]

    n, bins, patches = plt.hist(
        data,
        weights=weights, 
        range=rang,
        histtype='step',
        color=[soz_color, n_soz_color],
        label=['SOZ','N_SOZ'],
        #stacked=True,
    )

    print('Sum of bar heights for Soz: {0}'.format(sum(n[0])))
    print('Sum of bars heights for N_Soz: {0}'.format(sum(n[1])))
    print('Total sum of bar heights soz+n_soz: {0}'.format(sum(n[0])+sum(n[1])))
    
    plt.legend(loc = 'lower right')

    # add a 'best fit' line
    #y = mlab.normpdf(bins, mu, sigma)
    #plt.plot(bins, y, 'r--')

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
    #plt.subplots_adjust(left=0.15)
    plt.show()


def main():
    connection = Connect.get_connection()
    db = connection.deckard_new
    hfo_collection = db.HFOs

    hfos = hfo_collection.find(filter = { 'type': '1', 'loc5': 'Hippocampus', 'slow':1 },
                               projection = ['soz', 'power_pk', 'freq_pk', 'duration'])

    feature_names = ['Duration', 'Spectral content', 'Power Peak']
    data = { f_name:dict(soz=[], n_soz=[]) for f_name in feature_names }

    for h in hfos:
        power_pk = float(mt.log10((h['power_pk'])))
        freq_pk = float(h['freq_pk'])
        duration = float(h['duration']) * 1000 #seconds to milliseconds

        soz = soz_bool(h['soz'])
        soz_key = 'soz' if soz else 'n_soz'

        data['Power Peak'][soz_key].append(power_pk)
        data['Spectral content'][soz_key].append(freq_pk)
        data['Duration'][soz_key].append(duration)

    for feature_name in data.keys():
        plot_histogram(feature_name, data[feature_name]['soz'], data[feature_name]['n_soz'])

if __name__ == "__main__":
    main()

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

    print('Data soz (first 10) for {f_name}: {data}'.format(f_name=feature_name, data=soz_data[:10]))
    print('Data n_soz (first 10) for {f_name}: {data}'.format(f_name=feature_name, data=n_soz_data[:10]))

    # example data
    #mu = 100 # mean of distribution
    #sigma = 15 # standard deviation of distribution

    data = [soz_data, n_soz_data]
    soz_color = 'r'
    n_soz_color = 'k'

    rang = (0, 0.4*1e7) if feature_name == 'Power Peak' else None 

    n, bins, patches = plt.hist(
        data, 
        range=rang,
        histtype='step',
        color=[soz_color, n_soz_color],
        label=['SOZ','N_SOZ'],
        #stacked=True,
        density=True
    )
    '''
    #This should be the same as the plt.hist above
    
    n, bins, patches = plt.hist(
        soz_data,
        range=rang, 
        histtype='step',
        color=soz_color,
        label='SOZ',
        #stacked=True,
        density=True,

    )
    n, bins, patches = plt.hist(
        n_soz_data, 
        range=rang,
        histtype='step',
        color=n_soz_color,
        label='N_SOZ',
        #stacked=True,
        density=True,
    )
    '''
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
        power_pk = float(h['power_pk'])
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

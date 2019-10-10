import math as mt
import numpy as np
from scipy.stats import circmean
from matplotlib import pyplot as plt
from pymongo import MongoClient

from astropy.stats import rayleightest
from astropy import units as u
from decimal import Decimal


class Connect(object):
	@staticmethod
	def get_connection():
		return MongoClient("mongodb://localhost:27017")

def main():
	connection = Connect.get_connection()
	db = connection.deckard_new
	HFOs = db.HFOs

	rose_plot(HFOs, angle_step=(np.pi /9))#20 degrees
	
	# Query 2 rose plots
	'''
	rose_plot(HFOs, angle_step, 'Hippocampus')
	rose_plot(collection,angle_step, 'Brodmann area 21') middle temporal gyrus.
	rose_plot(collection,angle_step, 'Brodmann area 28') entorhinal cortex 
	rose_plot(collection,angle_step, 'Brodmann area 34')entorhinal cortex 

	rose_plot(collection,angle_step, 'Brodmann area 35') perirhinal cortex.
	rose_plot(collection,angle_step, 'Brodmann area 36') perirhinal cortex.
	'''

def rose_plot(collection, angle_step):
	fig = plt.figure()
	fig.suptitle('Spike angles\' distribution in Fast RonS')
	for soz_str in ['0', '1']:
		count_by_group, mean_angle, pvalue, hfo_count= get_cluster_angles(collection=collection, amp_step=angle_step, soz_str=soz_str)
		angles = []
		values = []
		HFO_TYPES = ['RonO', 'RonS', 'Spikes', 'Fast RonO', 'Fast RonS', 'Sharp Spikes']
		print('Count by fase group \n')
		print(count_by_group)
		for k, v in count_by_group.items():
			angles.append(angle_step * float(k) + angle_step/2)
			values.append(v)

		axe = plt.subplot( int( '12'+ str( int(soz_str) + 1 ) ), polar=True )
		title = 'SOZ' if soz_str == '1' else 'NSOZ'
		axe.set_title(title, fontdict={'fontsize': 16}, pad=10)
		polar_bar_plot(fig, axe, angles, values, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count)

	plt.show()


#type 4 == fast ronO type 5 == fast Rons
def get_cluster_angles(collection, amp_step, soz_str='1'):
	angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
	docs = collection.find({ 'type': "5", 'spike': 1, 'intraop': '0', 'soz':soz_str})
	hfo_count = docs.count()
	print('HFO count {0}'.format(hfo_count))
	angles = []
	for doc in docs:
		angle = doc['spike_angle'] % (2 * np.pi)  # Normalizing to 0 -- 2 pi
		angles.append(angle)
		angle_group_id = mt.floor(angle / amp_step)
		angle_grouped[str(angle_group_id)] += 1  # increment count of group

	for k,v in angle_grouped.items():  #normalizing values
		r_value = round( (v/hfo_count) * 100, 2 ) #show them as relative percentages
		angle_grouped[k] = r_value

	mean_angle = mt.degrees(circmean(angles))
	pvalue = float(rayleightest(np.array(angles)*u.rad))# doctest: +FLOAT_CMP
	
	return angle_grouped, mean_angle, pvalue, hfo_count

def polar_bar_plot(fig, ax1, angles, values, mean_angle, pvalue, hfo_count):
	# Data
	theta = angles
	heights = values

	bars = ax1.bar(angles, 
				   heights,
				   align='center',
				   #color='xkcd:salmon',
				   color=plt.cm.magma(heights),
				   width=0.1,
				   bottom=0.0,
				   edgecolor='k',
				   alpha=0.5,
				   label='HFO count (%)')

	annot = ax1.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                    arrowprops=dict(arrowstyle="->"))	
	
	annot.set_visible(False)


	## Main tweaks
	#max_count = max(values)
	max_value = max(values)
	radius_limit = max_value + (10 - max_value % 10) #finds next 10 multiple
	
	# Angle ticks
	ax1.set_xticks( np.linspace(0, 2* np.pi, 18, endpoint=False))
	# Radius limits
	ax1.set_ylim(0, max_value)
	# Radius ticks
	ax1.set_yticks(np.linspace(0, radius_limit, 5))
	
	# Radius tick position in degrees
	#ax1.set_rlabel_position(135)

	# Additional Tweaks
	ax1.grid(True)
	ax1.legend( loc='upper right', fancybox=True, bbox_to_anchor=(1.05,1.05))

	info_txt = 'Total HFO count: {count}'.format(count=hfo_count)
	ax1.text(-0.15, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

	raleigh_txt = ('Rayleigh Test \n \n'
				   'P-value: {pvalue} \n'
				   'Mean: {mean}° \n').format(pvalue= "{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))
	ax1.text(-0.15, 0, raleigh_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

	def update_annot(bar):
	    x = bar.get_x()+bar.get_width()/2.
	    y = bar.get_y()+bar.get_height()
	    annot.xy = (x,y)
	    text = "{c}".format( c=y )
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

if __name__ == "__main__":
	main()  

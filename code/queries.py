import math as mt
import numpy as np
from scipy.stats import circmean
from matplotlib import pyplot as plt
from pymongo import MongoClient

from astropy.stats import rayleightest
from astropy import units as u
from decimal import Decimal

def main(angle_step):
	connection = Connect.get_connection()
	db = connection.deckard_new
	HFOs = db.HFOs

	count_unique_patients_zone(HFOs, 'Brodmann area 21')	
	# Query 2 rose plots
	# rose_plot(collection,angle_step, 'Brodmann area 21')
	
	#rose_plot(HFOs, angle_step, 'Hippocampus')
	'''
	rose_plot(collection,angle_step, 'Brodmann area 28')
	rose_plot(collection,angle_step, 'Brodmann area 34')
	rose_plot(collection,angle_step, 'Brodmann area 35')
	rose_plot(collection,angle_step, 'Brodmann area 36')'''

class Connect(object):
	@staticmethod
	def get_connection():
		return MongoClient("mongodb://localhost:27017")




def rose_plot(collection, angle_step, loc_name):

	count_by_group, step, mean_angle, pvalue, hfo_count= get_cluster_angles(collection=collection, amp_step=angle_step, loc5=loc_name)
	angles = []
	values = []
	print('{name}. Count by fase group \n'.format(name=loc_name))
	print(count_by_group)
	for k, v in count_by_group.items():
		angles.append(step * float(k))
		values.append(v)

	polar_bar_plot(angles, values, loc_name=loc_name, mean_angle=mean_angle, pvalue=pvalue, hfo_count=hfo_count)


#type 4 == fast ronO type 5 == fast Rons
def get_cluster_angles(collection, amp_step, loc5='$exists'):
	angle_grouped = {str(angle_group_id): 0 for angle_group_id in range(mt.floor((2 * np.pi) / amp_step))}
	docs = collection.find({'$and': [{'type': "5"}, {'spike': 1}, {'intraop': '0'}, {'soz':'1'}, {'loc5': loc5}]})
	hfo_count = docs.count()
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
	
	return angle_grouped, amp_step, mean_angle, pvalue, hfo_count

def polar_bar_plot(angles, values, loc_name, mean_angle, pvalue, hfo_count):
	# Data
	theta = angles
	heights = values

	# Get an axes handle/object
	fig = plt.figure()
	ax1 = plt.subplot(111, polar=True,)
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
	plt.grid(True)
	plt.legend( loc='upper right', fancybox=True, bbox_to_anchor=(1.35,1))
	plt.title("Fast RonS SOZ in {location}".format(location=loc_name, ), fontdict={'fontsize': 16}, pad=10)

	raleigh_txt = ('Rayleigh Test \n \n'
				   'P-value: {pvalue} \n'
				   'Mean: {mean}° \n').format(pvalue= "{:.2E}".format(Decimal(pvalue)), mean=round(mean_angle))

	fig.text(-0.35, 0, raleigh_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

	info_txt = 'Total HFO count: {count}'.format(count=hfo_count)
	fig.text(-0.35, .95, info_txt, bbox=dict(facecolor='grey', alpha=0.5), transform=ax1.transAxes)

	#fig.text(.5, .00, 'Rayleigh test p-value {pvalue}'.format(pvalue=pvalue), ha='center')

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
	plt.show()

def count_unique_patients_zone(HFOs, zone):
	# unique patients ids for the filter below
	hfos_in_zone = HFOs.find({'$and': [{'intraop': '0'},{'loc5' : zone}]})
	docs = set()
	for doc in hfos_in_zone:
		docs.add(doc['patient_id'])
	patient_ids = list(docs)
	patient_ids.sort()
	print("Unique patients count in {0}: {1}".format(zone, len(patient_ids)))
	print(patient_ids)

if __name__ == "__main__":
	main(angle_step=(np.pi /9))  #20 degrees

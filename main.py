""" Group 2, Planning & Reinforcement Learning 2019
    Michael Accetto, Phillip Kersten
    Assignment Part 2"""



import numpy as np
import random
import pandas as pd
import env
from learning import learning # sorry about that
from viz import *
"""
Usage notes:

## Boltzmann exploration

	- takes one more argument (taus) which can be a list or float (between 0 and 1)
	examples:
		q_boltzmann = learning('BOLTZMANN')
		q_boltzmann(taus = [0.9, 0.5, 0.1], alpha, gamma, epsilon)
	or:
		q_boltzmann = learning('BOLTZMANN')
		q_boltzmann(taus = 0.3, alpha, gamma, epsilon)

## 

"""

if __name__ == "__main__":
	# can take 'Q', 'Q-ER', 'Q-ET' or 'SARSA'
	#methods = ["Q"]
	record = []
	q_learning = learning("BOLTZMANN")
	for series in range(30):

		log = q_learning() # epi=500 , alpha=.1 , gamma=.6 , epsilon=.05
		record.append(log)
	
	record = np.array(record)

	# let's plot
	avg, err = stat_ts(record)
	plot_ts(avg)

	import ipdb; ipdb.set_trace()

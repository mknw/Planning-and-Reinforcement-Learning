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
	methods = ["Q", "BOLTZMANN",  'Q-ER', 'Q-ET', 'SARSA', 'QQ']
	
	# parameters:
	"""
	* episodes: 1000
	* alpha: .1
	* gamma: .6
	* epsilon: .2
	* tau(s): [.6, .2, .05]
	* lambda: .3
	* 
	"""

	for method in methods:
		record = []
		brain = learning(method)
		for series in range(30):
	
			log = brain(episodes=1005) # epi=500 , alpha=.1 , gamma=.6 , epsilon=.05
			record.append(log)
		avg = stat_ts(np.array(record))
		if method == "Q":
			method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
		elif method == "BOLTZMANN":
			method_pars = r"$\alpha = 0.1, \gamma= 0.6, \tau = 0.3$"
		elif method == "Q-ER":
			method_pars = r"$\alpha = 0.1, \gamma= 0.6, \epsilon = 0.2$"
		elif method == "Q-ET":
			method_pars = r"$\alpha = 0.1, \gamma= 0.6, \epsilon = 0.3, \lambda = 0.3$"
		elif method == "SARSA":
			method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
		elif method == "QQ":
			method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
		
		
		plot(avg, method, method_pars, epsi=0.2, episodes=1000, save=True)

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

def test_params(method,iterations=10, **params):
	"""
	# parameters:
	## Q
	* alpha: .1; gamma: .6; epsilon: .2
	## BOLTZMANN
	* alpha: .1; gamma: .6; tau: 0.3
	## Q-ER (experience replay)
	* alpha: .1; gamma: .6; epsilon: .1
	## Q-ET (eligibility traces) 
	* alpha: .1; gamma: .6; epsilon: .2; lambda: .3
	## SARSA
	* alpha: .1; gamma: .6; epsilon: .2
	"""
	from itertools import product

	methods = ["Q", "BOLTZMANN",  'Q-ER', 'Q-ET', 'SARSA', 'QQ']

	if method not in methods:
		raise ValueError("'model' should be one of the following".format(*methods))
	
	lol = list(params.values()) # list of lists for each parameter


	param_combos = list(product(lol[0], lol[1], lol[2]))


	brain = learning(method)

	for param_set in param_combos:
		# unpack params
		p1, p2, p3 = param_set

		# make plot title with given params
		if method == "BOLTZMANN":
			method_pars = r"$\alpha = {}, \gamma = {}, \tau= {}$".format(p1, p2, p3)
		else:
			method_pars = r"$\alpha = {}, \gamma = {}, exploration \theta= {}$".format(p1, p2, p3)

		# start testing
		record = []
		for i in range(iterations):
			log = brain(p1, p2, p3)
			record.append(log)

		avg = stat_ts(np.array(record))
		fname = "method_a{}_g{}_e{}".format(p1, p2, p3)
		plot(avg, method, fname,method_pars, epsi=0.2, episodes=1000, save=True)
	# par_combos = [[i,j,k,] i for 
	# for series in range(5):

	# 	log = brain(episodes=1020



if __name__ == "__main__":
	# can take 'Q', 'Q-ER', 'Q-ET' or 'SARSA'

	methods = ["Q", "BOLTZMANN",  'Q-ER', 'Q-ET', 'SARSA', 'QQ']

	for m in methods:
		test_params(method=m, iterations=10, alpha=[.1,.2,.3],
					gamma=[.15,.2,.3], epsilon=[0.05,0.1,0.2])
	
	# for method in methods:
	# 	record = []
	# 	brain = learning(method)
	# 	for series in range(10):
	# 
	# 		log = brain(episodes=1005) # epi=500 , alpha=.1 , gamma=.6 , epsilon=.05

	# 		record.append(log)
	# 	
	# 	
	# 	avg = stat_ts(np.array(record))
	# 	if method == "Q":
	# 		method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
	# 	elif method == "BOLTZMANN":
	# 		method_pars = r"$\alpha = 0.1, \gamma= 0.6, \tau = 0.3$"
	# 	elif method == "Q-ER":
	# 		method_pars = r"$\alpha = 0.1, \gamma= 0.6, \epsilon = 0.2$"
	# 	elif method == "Q-ET":
	# 		method_pars = r"$\alpha = 0.1, \gamma= 0.6, \epsilon = 0.3, \lambda = 0.3$"
	# 	elif method == "SARSA":
	# 		method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
	# 	elif method == "QQ":
	# 		method_pars = r"$\alpha = 0.1, \gamma = 0.6, \epsilon = 0.2$"
	# 	
	# 	plot(avg, method, method_pars, epsi=0.2, episodes=1000, save=True)

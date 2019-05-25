""" Group 2, Planning & Reinforcement Learning 2019
    Michael Accetto, Phillip Kersten
    Assignment Part 2"""



import numpy as np
import random
import pandas as pd
from plot import plot
import env
from learning import learning # sorry about that

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
<<<<<<< HEAD
	methods = ["SARSA"]#, "SARSA", "Q-ET", "Q-ER"]
	plot_dict = {}
	all_all_plots = []
	for method in methods:
		plot_dict[method] = {}
		q_learning = learning(method)

		#for alpha in [.01,.05,.1,.2,.5]:
		all_plots = []
			# Record 100
		for i in range(10):
			plot_data = q_learning(alpha = .1, gamma = .6, epsilon = .2) # here alpha gamma and epsilon can be overwritten
			all_plots.append(plot_data)


		all_plots = np.array(all_plots)
		mean_plots = all_plots.mean(axis=0)
		sd_plots = all_plots.mean(axis=0)
		plot_dict[method]["all"] = all_plots
		plot_dict[method]["mean"] = mean_plots
		plot_dict[method]["sd"] = sd_plots
		df = pd.DataFrame(mean_plots)

		plot(mean_plots,method,epsi=.02,episodes=1000)
=======
	record = []
	q_learning = learning("BOLTZMANN")
	for series in range(5):

		log = q_learning() # epi=500 , alpha=.1 , gamma=.6 , epsilon=.05
		record.append(log)
	
	rr = np.array(record)

	import ipdb; ipdb.set_trace()
>>>>>>> 9995c09c91b70b9b0e4d817cf783fac463d7af68

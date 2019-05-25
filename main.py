""" Group 2, Planning & Reinforcement Learning 2019
    Michael Accetto, Phillip Kersten
    Assignment Part 2"""



import numpy as np
import random
import pandas as pd
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
# 	with open("learning.py") as fp:
# 		for i, line in enumerate(fp):
# 			if "\xe2" in line:
# 				print(i, repr(line))
# 				import ipdb; ipdb.set_trace()

	# can take 'Q', 'Q-ER', 'Q-ET' or 'SARSA'
	#methods = ["Q"]
	record = []
	q_learning = learning("BOLTZMANN")
	for series in range(5):

		log = q_learning() # epi=500 , alpha=.1 , gamma=.6 , epsilon=.05
		record.append(['series no: {}'.format(series), log])
	
	import ipdb; ipdb.set_trace()

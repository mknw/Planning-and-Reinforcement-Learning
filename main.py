""" Group 2, Planning & Reinforcement Learning 2019
    Phillip Kersten, Michael Accetto
    Assignment Part 1"""



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

	# can take 'Q', 'Q-ER', 'Q-ET', 'SARSA' or 'BOLTZMANN'
	q_learning = learning('BOLTZMANN')


	q_learning(taus = 0.5, alpha = .1, gamma = .6, epsilon = .1) # here alpha gamma and epsilon can be overwritten


	import ipdb; ipdb.set_trace()


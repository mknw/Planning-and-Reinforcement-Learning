""" Group 2, Planning & Reinforcement Learning 2019
    Phillip Kersten, Michael Accetto
    Assignment Part 1"""



import numpy as np
import random
import pandas as pd
import env
from learning import learning # sorry about that



if __name__ == "__main__":

	# can take 'Q', 'Q-ER', or 'SARSA'
	q_learning = learning('SARSA')


	q_learning( alpha = .1, gamma = .6, epsilon = .1) # here alpha gamma and epsilon can be overwritten


	import ipdb; ipdb.set_trace()



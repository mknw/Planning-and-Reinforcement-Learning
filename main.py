""" Group 2, Planning & Reinforcement Learning 2019
    Ilze Amanda Auzina, Phillip Kersten, Michael Accetto
    Assignment Part 1"""


RANDOM_POLICY_DISCOUNT_FACTOR   = 0.9
SLIP_CHANCE                     = 0.05
ACTION_PROBABILITY              = 0.95
OUTSIDE_GRID_PROBABILITY        = 1

ROBOT_STARTING_COORDINATES      = (3,0)
SHIPWRECK_COORDINATES           = (2,2)
GOAL_COORDINATES                = (0,3)
CRACK_COORDINATE_LIST           = [(1,1), (1,3), (2,3), (3,1), (3,2), (3,3)]

CRACK_REWARD                    = -10
SHIPWRECK_REWARD                =  20
GOAL_REWARD                     = 100



import numpy as np
import random
import pandas as pd
import env
from learning import learning # sorry about that



if __name__ == "__main__":

	# can take 'Q', 'Q-ER', or 'SARSA'
	q_learning = learning('Q')


	q_learning( alpha = .1, gamma = .6, epsilon = .1) # here alpha gamma and epsilon can be overwritten


	import ipdb; ipdb.set_trace()



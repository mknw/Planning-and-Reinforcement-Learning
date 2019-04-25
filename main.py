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
import env
from env import Environment, save_ts_pickle




if __name__ == "__main__":
	
	# Frozen lake environment
	FLenv = Environment()


	gamma = .9
	iterations = 1000

	#RANDOM POLICY
	#create a random policy
	policy_Random = np.ones([FLenv.observation_space_n, FLenv.action_space_n]) / FLenv.action_space_n

	V_randomPolicy, deltaIterations = FLenv.evaluate_policy(policy_Random, gamma)

	print("Done with Random Policy Evaluation!")
	print("can check delta for when it convergences")
	print(V_randomPolicy)
	print("")


	#VALUE ITERATION
	V_s_ValueIteration, bestMoves = FLenv.value_iteration(gamma, iterations)
	print("Done with Value Iteration")
	print(V_s_ValueIteration)
	print(bestMoves)
	print("")



	#POLICY ITERATION

	best_policy, Vs_Policy_Iteration = FLenv.policy_iteration(gamma)
	print("Done with Policy Iteration")
	print(best_policy)
	print("")
	print(Vs_Policy_Iteration)


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
import time
import random
import env
from env import Environment, save_ts_pickle
from pulp import *
import matplotlib.pyplot as plt
from evo import  evo




if __name__ == "__main__":
	
	# Frozen lake environment
	FLenv = Environment()


	gamma = .9

	#RANDOM POLICY
	#create a random policy

	policy_Random = np.ones([FLenv.observation_space_n, FLenv.action_space_n]) / FLenv.action_space_n

	start_random = time.time()
	V_randomPolicy, deltaRandom, iterationCount = FLenv.evaluate_policy(policy_Random, gamma)
	end_random=time.time()

	print("Done with Random Policy Evaluation!")
	print("time:" , end_random-start_random)
	print("can check delta for when it convergences")
	print(V_randomPolicy)
	print(iterationCount)
	print("")


	#VALUE ITERATION
	start_value = time.time()
	V_s_ValueIteration, bestMoves, deltaValue, iterationResult = FLenv.value_iteration(gamma)
	end_value = time.time()
	print("Done with Value Iteration")
	print("time:" , end_value-start_value)
	print(V_s_ValueIteration)
	print(bestMoves)
	print(iterationResult)
	print("")



	#POLICY ITERATION (HOWARD)
	start_policy = time.time()
	best_policy, Vs_Policy_Iteration, iterationsRun = FLenv.policy_iteration(gamma, method="howard")
	end_policy= time.time()

	print("Done with Howard Policy Iteration")
	print("time:", end_policy-start_policy)
	print(best_policy)
	print("")
	print(Vs_Policy_Iteration)
	print(iterationsRun)



	#POLICY ITERATION (SIMPLE)
	start_policy = time.time()
	#best_policy, Vs_Policy_Iteration, iterationsRun = FLenv.simple_policy(gamma)#FLenv.policy_iteration(gamma, method="simple")
	best_policy, Vs_Simple_Policy_Iteration, iterationsRun = FLenv.policy_iteration(gamma, method="simple")
	end_policy= time.time()
	print("Done with Simple Policy Iteration")
	print("time:", end_policy-start_policy)
	print("best policy:\n",best_policy)
	print("")
	print("Vs_Policy_Iteration\n",Vs_Policy_Iteration)
	print("iterationsRun\n",iterationsRun)

	#LINEAR PROGRAMMIN SOLUTION
	#define problem
	problem = LpProblem("MDP", LpMinimize)

	#set variable
	stateValue=LpVariable.dicts("StateValue", list(range(16)))

	ValuesAction = FLenv.ValueEveryAction(gamma)

	#constraints
	c01 = stateValue[0] >= ValuesAction[0][0]
	c02 = stateValue[0] >= ValuesAction[0][1]
	c03 = stateValue[0] >= ValuesAction[0][2]
	c04 = stateValue[0] >= ValuesAction[0][3]
	c10 = stateValue[1] >= ValuesAction[1][0]
	c11 =  stateValue[1] >= ValuesAction[1][1]
	c12 = stateValue[1] >= ValuesAction[1][2]
	c13 = stateValue[1] >= ValuesAction[1][3]
	c20 = stateValue[2] >= ValuesAction[2][0]
	c21 = stateValue[2] >= ValuesAction[2][1]
	c22 =  stateValue[2] >= ValuesAction[2][2]
	c23 =  stateValue[2] >= ValuesAction[2][3]
	c40 = stateValue[4] >= ValuesAction[4][0]
	c41 = stateValue[4] >= ValuesAction[4][1]
	c42 = stateValue[4] >= ValuesAction[4][2]
	c43 = stateValue[4] >= ValuesAction[4][3]
	c60 = stateValue[6] >= ValuesAction[6][0]
	c61 = stateValue[6] >= ValuesAction[6][1]
	c62 = stateValue[6] >= ValuesAction[6][2]
	c63 = stateValue[6] >= ValuesAction[6][3]
	c80 = stateValue[8] >= ValuesAction[8][0]
	c81 = stateValue[8] >= ValuesAction[8][1]
	c82 = stateValue[8] >= ValuesAction[8][2]
	c83 = stateValue[8] >= ValuesAction[8][3]
	c90 = stateValue[9] >= ValuesAction[9][0]
	c91 = stateValue[9] >= ValuesAction[9][1]
	c92 = stateValue[9] >= ValuesAction[9][2]
	c93 = stateValue[9] >= ValuesAction[9][3]
	c100 = stateValue[10] >= ValuesAction[10][0]
	c101 = stateValue[10] >= ValuesAction[10][1]
	c102 = stateValue[10] >= ValuesAction[10][2]
	c103 = stateValue[10] >= ValuesAction[10][3]
	c120 = stateValue[12] >= ValuesAction[12][0]
	c121 = stateValue[12] >= ValuesAction[12][1]
	c122 = stateValue[12] >= ValuesAction[12][2]
	c123 = stateValue[12] >= ValuesAction[12][3]

	# define objective function
	problem += stateValue[0] + stateValue[1] + stateValue[2] + stateValue[4] + stateValue[6] + stateValue[8] + stateValue[9] + stateValue[10] + stateValue[12]


#add contraints to the problem
	problem += c01
	problem += c02
	problem += c03
	problem += c04
	problem += c10
	problem += c11
	problem += c12
	problem += c13
	problem += c20
	problem += c21
	problem += c22
	problem += c23
	problem += c40
	problem += c41
	problem += c42
	problem += c43
	problem += c60
	problem += c61
	problem += c62
	problem += c63
	problem += c80
	problem += c81
	problem += c82
	problem += c83
	problem += c90
	problem += c91
	problem += c92
	problem += c93
	problem += c100
	problem += c101
	problem += c102
	problem += c103
	problem += c120
	problem += c121
	problem += c122
	problem += c123


	#solving
	start_LS = time.time()
	problem.solve()
	end_LS=time.time()

	#solution
	print("")
	print("Solution for Linear Programming")
	print("")
	for i in range(16):
		print(f"State {i}: {stateValue[i].varValue}")

	print("")
	print("time:", end_LS-start_LS)

	# Genetic algorithm
	V_s_evo, numit = evo()
	title =["Random policy","Value iteration","Howards Policy iteration", "Simple policy iteration", "Evolutionary algorithm"]
	for i,x in enumerate([V_randomPolicy,V_s_ValueIteration,Vs_Policy_Iteration,Vs_Simple_Policy_Iteration, V_s_evo]):
		plt.figure(1)
		plt.title(title[i])
		tb = plt.table(cellText=x, loc=(0, 0), cellLoc='center')
		tc = tb.properties()['child_artists']
		ax = plt.gca()
		ax.set_xticks([])
		ax.set_yticks([])
		plt.savefig(title[i]+".png", dpi=300)

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



	#RANDOM POLICY
	#simplify by making V_s an array
	V_s = np.zeros((FLenv.size, FLenv.size))
	states = [12, 1, 2, 4, 6, 8, 9, 10, 0] #non terminal states
	actions = [i for i in range(4)]
	gamma = .9

	iterations = 1000
	log = []

	#records delta change after each iteration
	deltaAll=[]
	for i in range(iterations):
	#	FLenv.reset()
	#	epochs, penalties, tot_reward = 0, 0, 0
	#	done = False
	#	print("Iteration: " + str(i))
	#	while not done:
		deltaIteration=[]
		copyV_s=np.copy(V_s)
		for s in states: #s is an integer
			stt_val = 0
			state = np.zeros((16)).astype(bool) #boolean array
			state[s] = True
			for act in actions: #act is an integer

				s_prime, reward, hit_grid = FLenv.sim_step(state, act) #pass a boolean array and an int

				if hit_grid:
					stt_val += (.25 * (reward+(gamma*V_s[s_prime[0][0],s_prime[0][1]])))
				else:
					stt_val += (.25 * .95 * (reward+(gamma*V_s[s_prime[0][0],s_prime[0][1]])))
			#list of delta value fro every state in an itteration
			deltaIteration.append(np.abs(V_s[np.reshape(state, (4, 4)) ]- stt_val))
			V_s[ np.reshape(state, (4, 4)) ] = stt_val
		deltaAll.append(deltaIteration)
			# Current State fetched from Env object as 16values long 1-hot vector.
		if i in [0, 1, 2, 9, 99, iterations - 1]:
			print("Iteration {}".format(i + 1))
			print(V_s)
			print("")

	print("Done with Random Policy Evaluation!")
	print("can check delta for when it convergences")
	print("")

	delta_total=[]
	Vs=np.zeros((FLenv.size, FLenv.size))
	dict={}
	for i in states:
		dict[i]=0

	#VALUE ITERATION
	for k in range(iterations):
		delta_valueit=[]
		for s in states:  # s is an integer
			state = np.zeros((16)).astype(bool)  # boolean array
			state[s] = True
			all_actions_for_s = FLenv.actions(state,Vs)
			best_action=np.max(all_actions_for_s)
			index_action=list(all_actions_for_s).index(best_action)
			action_letter=FLenv.map_actions[index_action]
			#index_action=np.where(all_actions_for_s==best_action)
			#record best action for a certain state
			dict[s]=action_letter

			#update Vs
			delta_valueit.append(np.abs(Vs[np.reshape(state, (4, 4))] - best_action))
			Vs[np.reshape(state, (4, 4))] = best_action
		delta_total.append(delta_valueit)

		if k in [0, 1, 2, 9, 99, iterations - 1]:
			print("Iteration {}".format(k + 1))
			print(Vs)
			print("")

	print("Done with Value Iteration")
	print(dict)


	#
			# Bellman Equation:



			# append to log
		#	epochs += 1
		
		#log.append([i, epochs, penalties, tot_reward])
		
		#if i % 100 == 0:

		#	print("Episode: {i}.")

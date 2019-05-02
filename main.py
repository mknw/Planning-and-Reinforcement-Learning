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



def policy_iter(environment, epochs=1000):
	#RANDOM POLICY
	#simplify by making V_s an array
	V_s = np.zeros((environment.size, environment.size)) #
	states = [12, 1, 2, 4, 6, 8, 9, 10, 0] #non terminal states
	actions = [i for i in range(4)]
	gamma = .9
	
	########## HEAD
	reward_plot = {}
	reward_plot["single"] = []
	reward_plot["cumulative"] = []
	log = []

	deltaAll = list()
	for i in range(epochs):
		environment.reset()
		penalties, tot_reward = 0, 0
		done = False

		deltaIteration = list()
		copyV_s = np.copy(V_s)
		for s in states:
			stt_val = 0
			state = np.zeros((16)).astype(bool)
			state[s] = True
			temp = []
			for act in actions:

	
				next_state, reward, done = environment.sim_step(state, act)
	
				if environment.movement_out_grid:
					stt_val += (.25 * (reward+(gamma*V_s[environment.pos_mtx])))
				else:
					stt_val += (.25 * .95 * (reward+(gamma*V_s[environment.pos_mtx])))

				if reward  == -10:
					penalties += 1
				tot_reward += reward

				temp.append(reward)

				reward_plot["single"].append(reward)
				reward_plot["cumulative"].append(tot_reward)

			# list of delta value from every state in an iteration
			deltaIteration.append( np.abs( V_s[np.reshape(state, (4, 4))] - stt_val))
			# assign state value over all actions: 
			V_s[ np.reshape(state, (4, 4)) ] = stt_val
		deltaAll.append(deltaIteration)
	
		log.append([i, penalties, tot_reward, V_s])

		if i % 100 == 0:
			print("Epoch: ", i, ".")
			print(V_s)
			print("MEAN:  ",np.mean(V_s))
	return (log, deltaAll, reward_plot)


def value_iter(environment, epochs=1000):

	delta_total = list()
	Vs = np.zeros( (environment.size, environment.size) )
	states = [12, 1, 2, 4, 6, 8, 9, 10, 0] #non terminal states
	dict_={}
	for i in states:
		dict_[i]=0

	#VALUE ITERATION
	for k in range(epochs):
		delta_valueit=[]
		for s in states:  # s is an integer
			state = np.zeros((16)).astype(bool)  # boolean array
			state[s] = True
			action_outcome = np.zeros(4)

			for act in range(4):
				s_prime, reward, done = environment.sim_step(state, act)
				if environment.movement_out_grid:
					action_outcome[act] += reward+(0.9*Vs[environment.pos_mtx])
				else:
					action_outcome[act] += 0.95*(reward+(0.9*Vs[environment.pos_mtx]))

			all_actions_for_s = action_outcome
			best_action = np.max(all_actions_for_s)
			index_action = list(all_actions_for_s).index(best_action)
			action_letter = environment.map_actions[index_action]
			#index_action=np.where(all_actions_for_s==best_action)
			#record best action for a certain state
			dict_[s]=action_letter

			#update Vs
			delta_valueit.append(np.abs(Vs[np.reshape(state, (4, 4))] - best_action))
			Vs[np.reshape(state, (4, 4))] = best_action
		delta_total.append(delta_valueit)

		if k in [0, 1, 2, 9, 99, epochs- 1]:
			print("Iteration {}".format(k + 1))
			print(Vs)
			print("MEAN:     ",np.mean(Vs))

	print("Done with Value Iteration")
	print(dict_)

	return (Vs, log)


if __name__ == "__main__":
	
	try:
		import os
		save_dir = "howard-trials"
		os.mkdir(save_dir)
	except FileExistsError:
		pass

	# Frozen lake environment
	FLenv = Environment()
	# save timestamped log file

	log, deltaAll, plot_data = policy_iter(FLenv, epochs=1000)

	save_path = os.path.join(save_dir, "pol_iter_log")
	save_ts_pickle(save_path, log)

	log, deltaAll = value_iter(FLenv, epochs=1000)

	save_path = os.path.join(save_dir, "val_iter_log")
	save_ts_pickle(save_path, log)


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
	 
	V_s = np.zeros([FLenv.size, FLenv.size])
	states = [0, 1, 2, 4, 6, 8, 9, 10, 12]
	actions = [i for i in range(4)]
	gamma = .9

	episodes = 1000 
	
	log = []

	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		while not done:

			for s in states:
				stt_val = 0
				state = np.zeros((16)).astype(bool)
				state[s] = True
				for act in actions:

					next_state, reward, done = FLenv.sim_step(state, act)

					if FLenv.out_grid:
						stt_val += (.25 * (reward+(gamma*V_s[FLenv.pos_mtx])))
					else:
						stt_val += (.25 * .95 * (reward+(gamma*V_s[FLenv.pos_mtx])))
				V_s[ np.reshape(state, (4, 4)) ] = stt_val
			# Current State fetched from Env object as 16values long 1-hot vector.

			if reward == -10:
				penalties += 1
			tot_reward += reward 
			
			# Bellman Equation:
			

			# append to log
			epochs += 1
		
		log.append([i, epochs, penalties, tot_reward])
		
		if i % 100 == 0:
			print("Episode: {i}.")

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
	 
	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])
	
	alpha, gamma, epsilon = .1, .6, .1
	episodes = 1000 
	
	log = []

	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		while not done:
			
			if random.uniform(0, 1) < epsilon or i<=250:
				action = FLenv.sample_action()
				C_S = FLenv.pos_mtx.flatten().astype(bool) 
			else:
				# C(urrent) S(tate)
				C_S = FLenv.pos_mtx.flatten().astype(bool) 
				action = np.argmax(Q[C_S])

			next_state, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward 
			
			# Bell's Equation:
			prev_val = Q[C_S, action]
			next_max = np.max(Q[next_state])
			new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
			# update Q table.
			Q[C_S, action] = new_val

			state = next_state
			# append to log
			epochs += 1
		
		log.append([i, epochs, penalties, tot_reward])
		
		if i % 100 == 0:
			print("Episode: {i}.")
	


	save_ts_pickle('log', log)
	save_ts_pickle('Qtable', Q)

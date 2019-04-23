""" Group 2, Planning & Reinforcement Learning 2019
    Ilze Amanda Auzina, Phillip Kersten,
    Florence van der Voort, Stefan Wijtsma
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
from env import Environment




if __name__ == "__main__":
	
	# Frozen lake environment 
	FLenv = Environment()
	 
	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])
	
	alpha, gamma, epsilon = .1, .6, .1

	
	for i in range(1000):
		FLenv.reset()
		epochs, penalties, reward = 0, 0, 0
		done = False

		while not done:
			if random.uniform(0, 1) < epsilon:
				action = FLenv.sample_action()
			else:
				# C(urrent) S(tate)
				C_S = FLenv.pos_mtx.flatten().astype(bool) 
				action = np.argmax(Q[C_S])

			next_state, reward, done = FLenv.step(action)
			
			prev_val = Q[C_S, action]
			next_max = np.max(Q[next_state])

			new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
			Q[C_S, action] = new_val

			state = next_state
			epochs += 1
		
		if i % 100 == 0:
			print("Episode: {i}.")


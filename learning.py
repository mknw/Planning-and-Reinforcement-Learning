import numpy as np
import random
import pandas as pd
import env
from env import Environment, save_ts_pickle

def q_learning(alpha, gamma, epsilon):

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

			if random.uniform(0, 1) < epsilon or i<=250: # change to: i<=episodes to turn on random policy.
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
				C_S = FLenv.pos_mtx.flatten().astype(bool)
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
				C_S = FLenv.pos_mtx.flatten().astype(bool)
				action = np.argmax(Q[C_S])

			next_state, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# Bellman Equation:
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
	save_ts_pickle('Qlog', log)
	save_ts_pickle('Qtable', Q)
	


def q_learning_er(alpha, gamma, epsilon):
	FLenv = Environment()
	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])
	episodes = 1000
	log = []
	# D is the database for experience replay
	D = []
	# number of replay N
	N = 10
	# Time-step k
	k = 0
	# index of transition sample
	l = 1
	# length of trajectory for experience replay (how many timesteps should be performed before sampling from D?)
	T = 50
	# initial state
	state = FLenv.get_state()
	all_rewards = []
	for i in range(episodes):
		reward_per_step = []
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		while not done:
			# Either...
			if random.uniform(0, 1) < epsilon or i<=250: # change to: i<=episodes to turn on random policy.
				# ...choose random action...
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
				C_S = FLenv.pos_mtx.flatten().astype(bool)
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
				C_S = FLenv.pos_mtx.flatten().astype(bool)
				# ...or best action
				action = np.argmax(Q[C_S])

			# Observe next state and reward
			next_state, reward, done = FLenv.step(action)

			reward_per_step.append(reward)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# compute transition index with current trajectory:
			t = (k-(l-1))*T

			# add transition sample to the database D
			#         1   2      3          4
			sample = (C_S,action,next_state,reward)
			D.append(sample)

			# Increment state
			state = next_state

			# Update k
			k = k+1
			# every T steps update q function with experience
			if k == l*T:
				n_iter = N*l*T
				for i in range(n_iter):
					# Randomly choose an experience from D
					index = int(random.uniform(0, len(D)))
					C_S, action, next_state, reward = D[index]

					# Bellman Equation:
					prev_val = Q[C_S, action]
					next_max = np.max(Q[next_state])
					new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
					# update Q table.
					Q[C_S, action] = new_val

				# Update l
				l = l+1

			# append to log
			epochs += 1
		all_rewards.append(reward_per_step)

		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")

	save_ts_pickle('ER-log', log)
	save_ts_pickle('ER-Qtable', Q)


def sarsa(alpha, gamma, epsilon):
	
	## define sarsa
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

			if random.uniform(0, 1) < epsilon or i<=250: # change to: i<=episodes to turn on random policy.
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
				C_S = FLenv.pos_mtx.flatten().astype(bool)
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
				C_S = FLenv.pos_mtx.flatten().astype(bool)
				action = np.argmax(Q[C_S])
			
			# take action A, observe R, S_prime
			next_state, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# SARSA
			V = Q[C_S, action] # save Q(S, A)
			A_prime = np.max(Q[next_state]) # choose A' from S' using policy after Q
			new_V = V + alpha * (reward + (gamma * A_prime) - V)
			Q[C_S, action] = new_V
			

			state = next_state
			action = A_prime
			# append to log
			epochs += 1

		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")
	save_ts_pickle('SARSA-log', log)
	save_ts_pickle('SARSA-Qtable', Q)




def learning(method):
	""" wrapper function for:
	* Q-learning
	* Q-learning with experience replay
	* SARSA
	"""

	if method == "Q":
		return q_learning

	if method == "Q-ER":
		return q_learning_er

	if method == "SARSA":
		return sarsa

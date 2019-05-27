# -*- coding: utf-8 -*-

import numpy as np
import random
from copy import copy
import pandas as pd
import env
from env import Environment, save_ts_pickle


def q_learning(alpha = .1, gamma = .6, epsilon = .2):
	"""
	Basic implementation of Q learning algorithm.
	Takes:
		1. alpha: step size parameter
		2. gamma: discount rate parameter
		3. epsilon: epsilon-greedy policy parameter
	"""
	episodes = 1020
	# Frozen lake environment
	FLenv = Environment()

	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])


	log = []
	plot_data = []

	for i in range(episodes):
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:

			if random.random() < epsilon: # change to: i<=episodes to turn on random policy.
				action = FLenv.sample_action()
			else:
				action = np.argmax(Q[C_S])

			s_prime, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# Bellman Equation:
			prev_val = Q[C_S, action]
			next_max = np.max(Q[s_prime])
			new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
			# update Q table.
			Q[C_S, action] = new_val
			C_S = s_prime
			epochs += 1
		# prepare next episode
		FLenv.reset()
		log.append([i, epochs, penalties, tot_reward])
		plot_data.append(tot_reward)

		if i % 100 == 0:
			print("Episode: {i}.")
	# save_ts_pickle('Qlog', log)
	# save_ts_pickle('Qtable', Q)
	return log


def q_boltzmann(episodes=1000, alpha = .1, gamma = .6, taus = [0.3]):
	# local import
	import math

	# Frozen lake environment
	FLenv = Environment()
	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])
	episodes = 1020
	log = []
	

	if isinstance(taus, list):
		if len(taus) > 1:
			anneal_point = episodes / len(taus)
		else:
			tau = taus[0]
			anneal_point = False
	elif isinstance(taus, float):
		tau = taus
	else:
		raise Exception

	### boltzmann exploration
	def softmax(tau, actions):
		actions = actions[0]
		# import ipdb; ipdb.set_trace()
		tot = sum([math.exp(v/tau) for v in actions])
		probs = [math.exp(v/tau)/ tot for v in actions]
		threshold = random.random()
		cum_prob = float(0)
		# 
		for i in range(len(probs)):
			cum_prob += probs[i]
			if cum_prob > threshold:
				return i
		return np.argmax(probs)
	###

	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False

		if anneal_point:
			if i % anneal_point == 0:
				idx = i // anneal_point
				tau = taus[idx]

		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:

			action = softmax(tau, Q[C_S])
			s_prime, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# Bellman Equation:
			prev_val = Q[C_S, action]
			next_max = np.max(Q[s_prime])
			new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
			# update Q table.
			Q[C_S, action] = new_val
			C_S = s_prime
			epochs += 1
		# prepare next episode
		FLenv.reset()

		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")
	# save_ts_pickle('QBOLT-log', log)
	# save_ts_pickle('QBOLT-table', Q)
	return log



def q_learning_er(alpha = .1, gamma = .6, epsilon = .1):
	# Frozen lake environment
	FLenv = Environment()

	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])


	episodes = 1020

	log = []

	# D is the database for experience replay
	D = []
	# number of replay N
	N = 5
	# Time-step k
	k = 0
	# index of transition sample
	l = 1
	# length of trajectory for experience replay (how many timesteps should be performed before sampling from D?)
	T = 50
	# initial state
	state = FLenv.get_state()
	all_rewards = []
	acc_reward = 0.0
	for i in range(episodes):
		reward_per_step = []
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:
			# Either...
			if random.uniform(0, 1) < epsilon:# or i <= 250:  # change to: i<=episodes to turn on random policy.
				# ...choose random action...
				action = FLenv.sample_action()
			else:
				# ...or best action
				action = np.argmax(Q[C_S])

			# Observe next state and reward
			next_state, reward, done = FLenv.step(action)

			# Something with penalties
			if reward == -10:
				penalties += 1
			tot_reward += reward

			# compute transition index with current trajectory:
			t = (k - (l - 1)) * T

			# add transition sample to the database D
			#         1:state  2:action  3: next state  4: reward
			sample = (C_S,     action,   next_state,    reward)
			D.append(sample)

			# Increment state
			C_S = next_state

			# Update k
			k = k + 1
			# every T steps update q function with experience
			if k == l * T:
				# Iterate NlT times
				n_iter = N * l * T
				for i in range(n_iter):

					# Randomly choose an experience from D
					index = int(random.uniform(0, len(D)))
					C_S, action, next_state, reward = D[index]

					# Bellman Equation:
					prev_val = Q[C_S, action]
					next_max = np.max(Q[next_state])
					new_val = (1 - alpha) * prev_val + alpha * (reward + gamma * next_max)
					# update Q table.
					Q[C_S, action] = new_val

				# Update l (number or replays played)
				l = l + 1

			C_S = next_state
			epochs += 1
		all_rewards.append(tot_reward)

		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")
	#save_ts_pickle('ER-log', log)
	#save_ts_pickle('ER-Qtable', Q)
	return log


def q_learning_et(alpha = .1, gamma = .6, epsilon = .2, lmbda = 0.3):
	# Initialize decay rate $\lambda$
	"""
	Eligibility traces: replacing traces

	source:
	S. Singh and R. Sutton. Reinforcement learning with replacing eligibility traces.
	Machine Learning, 22:123–158, 1996.
	"""

	# Frozen lake environment
	FLenv = Environment()

	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])

	episodes = 1020

	log = []

	# Initialize eligibility trace
	E = [0]


	# Initialize counter
	k = 0

	all_reward = []


	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:
			if random.uniform(0, 1) < epsilon:  # change to: i<=episodes to turn on random policy.

				# whenever an exploratory action is taken, the
				# causality of the sequence of state-action pairs is broken and
				# the eligibility trace should be reset to 0.
				E[k] = 0
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
				action = np.argmax(Q[C_S])

			next_state, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# Updating counter
			k = k+1
			# Updating the trace
			next_max = np.max(Q[next_state])
			E.append(min(lmbda*E[k-1] + (reward + gamma * next_max), 1))
			# Bellman Equation:
			prev_val = Q[C_S, action]
			new_val = (1 - alpha) * prev_val + alpha * E[k]
			# update Q table.
			Q[C_S, action] = new_val

			C_S = next_state
			# append to log
			epochs += 1

		# Update
		all_reward.append(tot_reward)
		log.append([i, epochs, penalties, tot_reward])

		# if i % 100 == 0:
		# 	print("Episode: {i}.")

	#save_ts_pickle('Q-ET-log', log)
	#save_ts_pickle('Q-ET-Qtable', Q)
	return log

def sarsa(alpha = .1, gamma = .6, epsilon = .2):
	
	## define sarsa
	FLenv = Environment()

	Q = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])

	episodes = 1020

	log = []

	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:

			if random.uniform(0, 1) < epsilon: # change to: i<=episodes to turn on random policy.
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
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
			

			C_S = next_state
			action = A_prime
			# append to log
			epochs += 1
		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")
	return log


def q_q_learning(alpha = .1, gamma = .6, epsilon = .2):

	# Frozen lake environment
	FLenv = Environment()

	Q1 = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])
	Q2 = np.zeros([FLenv.observation_space_n, FLenv.action_space_n])

	episodes = 1020

	log = []

	for i in range(episodes):
		FLenv.reset()
		epochs, penalties, tot_reward = 0, 0, 0
		done = False
		print("episode: " + str(i))
		C_S = FLenv.pos_mtx.flatten().astype(bool)
		while not done:

			# Choose a* from table 1 or 2
			if random.uniform(0, 1) > 0.5:
				update="A"
			else:
				update="B"

			if random.uniform(0, 1) < epsilon: # change to: i<=episodes to turn on random policy.
				action = FLenv.sample_action()
				# Current State fetched from Env object as 16values long 1-hot vector.
			else:
				# C(urrent)S(tate) as 1-hot, 16 vals-long vector (same thing).
				# Choose a based on Q1,Q2
				if update == "A":
					action = np.argmax(Q1[C_S])

				if update == "B":
					action = np.argmax(Q2[C_S])

			next_state, reward, done = FLenv.step(action)

			if reward == -10:
				penalties += 1
			tot_reward += reward

			# Update Q with value from Q_other
			# Bellman Equation:
			if update == "B":
				prev_val = Q1[C_S, action]
				next_max = np.max(Q1[next_state])
				new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
				# update Q1 table.
				Q1[C_S, action] = new_val

			if update == "A":
				prev_val = Q2[C_S, action]
				next_max = np.max(Q2[next_state])
				new_val = (1-alpha)*prev_val+alpha*(reward + gamma * next_max)
				# update Q1 table.
				Q2[C_S, action] = new_val

			C_S = next_state
			# append to log
			epochs += 1

		log.append([i, epochs, penalties, tot_reward])

		if i % 100 == 0:
			print("Episode: {i}.")
	return log


def learning(algo):
	""" wrapper function for:
	* Q-learning
	* Q-learning with experience replay
	* SARSA
	"""
	methods = ["Q", "SARSA", "Q-ET", "Q-ER", "BOLTZMANN","QQ"]


	if algo not in methods:
		raise ValueError("Method not found.")


	if algo == "Q":
		return q_learning

	if algo == "Q-ER":
		return q_learning_er

	if algo == "Q-ET":
		return q_learning_et

	if algo == "SARSA":
		return sarsa

	if algo == "BOLTZMANN":
		return q_boltzmann

	if algo == "QQ":
		return q_q_learning

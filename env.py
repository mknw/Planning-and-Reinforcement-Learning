import numpy as np
import curses
import random

""" rewards memo """
# ship wreck -> +20
# goal -> +100
# crack -> -10 and end episode.
"""" probs memo """


# moving: 0.95
# slipping: 0.05


class Environment(object):

	def __init__(self, size=4, start_coords=(3, 0), wreck_coords=(2, 2), goal_coords=(0, 3),
				 crack_coords=[(1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)],
				 non_terminal_states=[0, 1, 2, 4, 6, 8, 9, 10, 12],
				 actions=[i for i in range(4)], threshold=0.00001):
		self.size = size

		self.start_coords = start_coords
		self.wreck_coords = wreck_coords
		self.goal_coords = goal_coords
		self.crack_coords = crack_coords
		self.states = non_terminal_states
		self.actions = actions
		self.threshold = threshold

		self.lake_map = np.zeros((size, size)).astype(str)
		self.state_space = ["S", "F", "C", "W", "G"]  # start, frozen, crack, wreck, goal
		self.action_space = ['U', 'D', 'R', 'L']  # up, down, right, left
		self.reset()
		self.observation_space_n = self.lake_map.size
		self.action_space_n = len(self.action_space)

		self.map_actions = {k: v for k, v in enumerate(self.action_space)}
		pass

	def __repr__(self):
		# display frozen lake_map on terminal.
		return str(self.lake_map)

	def sample_action(self):
		# choose random action from self, action_space
		# (randomly called against epsylon)
		# return: action
		return np.random.randint(0, len(self.action_space))

	def reset(self):

		# reset enviroment to pristine conditions.
		self.lake_map[:] = "F"
		self.lake_map[self.start_coords] = "S"
		self.lake_map[self.wreck_coords] = "W"
		self.lake_map[self.goal_coords] = "G"
		# crack coordinates are many. We 'unzip' the tuples and index the matrix only once:
		y_crack, x_crack = list(zip(*self.crack_coords))
		self.lake_map[y_crack, x_crack] = "C"
		print("Environment renewed.")
		# generates self.pos_mtx to keep track of agent.
		self.get_pos(reset=True)  # set position matrix at "S"tart.
		self.out_grid = False
		pass

	def get_pos(self, reset=False):
		# get agent starting position
		if reset == True:
			self.pos = list(np.where(self.lake_map == "S"))
		# keep track of agent position by boolean pointer.
		self.pos_mtx = np.zeros((self.size, self.size)).astype(bool)
		self.pos_mtx[tuple(self.pos)] = True
		# print("Agent at position: " + str(self.pos))
		return self.pos_mtx

	def move(self, action_n):
		"""Performs actual movement.
		Takes action_n (0, 1, 2 or 3);
		Updates self.pos_mtx (state matrix);
		Returns state tile label (F, W, G or C)."""

		hit_grid = False
		action = self.map_actions[action_n]  # convert the integer to a letter

		# make a copy to check it moved outside the grid
		copy_s = np.copy(self.pos)
		# prev_pos = np.copy(self.pos)

		# converted action movements to coordinates as pos now is a coordinate
		# update pos to s_prime
		if action == "U":  # [-1,0]
			self.pos = np.array(self.pos) + [-1, 0]
		elif action == "D":  # [1,0]
			self.pos = np.array(self.pos) + [1, 0]
		elif action == "L":  # [0,-1]
			self.pos = np.array(self.pos) + [0, -1]
		elif action == "R":  # [0,1]
			self.pos = np.array(self.pos) + [0, 1]
		else:
			raise ValueError("Possible action values are: " + str(self.map_actions))

		# check if the agent moved oustide the grid
		# if true then adjust the value to the saved starting position (didnt move)
		if -1 in self.pos or 4 in self.pos:
			self.pos = copy_s
			hit_grid = True

		s_prime = self.lake_map[self.pos[0][0], self.pos[0][1]]  # where s_prime is a letter from the map
		# create state vector by flattening state map.
		return s_prime, hit_grid

	def step_normal(self, action):

		s_prime, hit_grid = self.move(action)  # move.

		action_probablity = 0.95

		if s_prime == "C":  # Did he move onto a crack?
			# print("GAME OVER")
			# done = True
			reward = -10
		# next_state = self.get_state()
		# return s_prime, reward, hit_grid # return here to stepside "self.render()" after agent loss.

		elif s_prime == "W":
			reward = 20

		elif s_prime == "G":  # Goal reached?
			# print("YOU WON!")
			# done = True  # episode ended.
			reward = 100
		else:
			reward = 0

		s_prime_coordinates = self.pos

		return s_prime_coordinates, reward, hit_grid, action_probablity

	def step_slide(self, action):
		slide_probability = 1
		s_prime, hit_grid = self.move(action)

		if not hit_grid:
			slide_probability = 0.05
			stop = False
			while not stop:
				s_prime, stop = self.move(action)
				if s_prime == "C":
					reward = -10
					stop = True
				elif s_prime == "G":
					reward = 100
					stop = True
				else:
					reward = 0

		else:
			reward = 0

		s_prime_coordinates = self.pos

		return s_prime_coordinates, reward, hit_grid, slide_probability

	def sim_step_normal(self, state, action):

		self.pos_mtx = np.reshape(state, (4, 4))  # format the boolan state array as a matrix
		# pos_mtx at this point contains the location of state s (NOT S_PRIME)
		# extract the position and seve in pos
		self.pos = np.argwhere(self.pos_mtx == True)

		s_prime, reward, hit_grid, action_probability = self.step_normal(action)  # here s_prime is coordinates
		return s_prime, reward, hit_grid, action_probability

	def sim_step_slide(self, state, action):
		self.pos_mtx = np.reshape(state, (4, 4))  # format the boolan state array as a matrix
		# pos_mtx at this point contains the location of state s (NOT S_PRIME)
		# extract the position and seve in pos
		self.pos = np.argwhere(self.pos_mtx == True)

		s_prime, reward, hit_grid, action_probability = self.step_slide(action)  # here s_prime is coordinates
		return s_prime, reward, hit_grid, action_probability

	def calc_action_value(self, state, Vs, gamma):
		action_outcome = np.zeros(4)
		for act in range(self.size):
			s_prime, reward, hit_grid, action_probablity = self.sim_step_normal(state, act)
			s_prime_slip, reward_slip, hit_grid_slip, action_probability_slip = self.sim_step_slide(state, act)
			if hit_grid:
				action_outcome[act] += reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]])
			else:
				action_outcome[act] += action_probablity * (reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]]))
				action_outcome[act] += action_probability_slip * (reward_slip + (gamma * Vs[s_prime_slip[0][0], s_prime_slip[0][1]]))

		return action_outcome

	def ValueEveryAction(self, gamma):  # V is a dict
		Vs = np.zeros((4, 4))

		Values = np.zeros((self.observation_space_n, self.size))


		while True:
			delta = 0
			for s in self.states:
				state = np.zeros((16)).astype(bool)  # convert to boolean array
				state[s] = True
				best_value = 0
				for act in range(self.size):
					value = 0
					s_prime, reward, hit_grid, action_probablity = self.sim_step_normal(state, act)
					s_prime_slip, reward_slip, hit_grid_slip, action_probability_slip = self.sim_step_slide(state, act)
					if hit_grid:
						value += reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]])
					else:
						value += action_probablity * (reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]]))
						value += action_probability_slip * (
									reward_slip + (gamma * Vs[s_prime_slip[0][0], s_prime_slip[0][1]]))

					if value > best_value:
						best_value = value

					Values[s][act] = value
				delta = max(delta, np.abs(Vs[np.reshape(state, (4, 4))] - best_value))
				Vs[np.reshape(state, (4, 4))] = best_value
			if delta <= self.threshold:
				return Values

	def evaluate_policy(self, policy, gamma):
		V_s = np.zeros((self.size, self.size))
		deltaAll = []
		iterations = 0

		while True:
			deltaIteration = []
			delta = 0
			for s in self.states:  # select a state s (s is an integer)
				expected_val = 0
				state = np.zeros((16)).astype(bool)  # convert to boolean array
				state[s] = True
				for act, act_probability in enumerate(
						policy[s]):  # select all possible actions in this state (act is an integer)

					s_prime, reward, hit_grid, action_probability = self.sim_step_normal(state,
																						 act)  # pass a boolean array and an int

					s_prime_slip, reward_slip, hit_grid_slip, action_probability_slip = self.sim_step_slide(state, act)

					if hit_grid:
						expected_val += (act_probability * (reward + (gamma * V_s[s_prime[0][0], s_prime[0][1]])))
					else:
						expected_val += (act_probability * action_probability * (
									reward + (gamma * V_s[s_prime[0][0], s_prime[0][1]])))
						expected_val += (act_probability * action_probability_slip * (
									reward_slip + (gamma * V_s[s_prime_slip[0][0], s_prime_slip[0][1]])))

				# list of delta value fro every state in an itteration
				delta = max(delta, np.abs(V_s[np.reshape(state, (4, 4))] - expected_val))
				deltaIteration.append(np.abs(V_s[np.reshape(state, (4, 4))] - expected_val))
				V_s[np.reshape(state, (4, 4))] = expected_val

			deltaAll.append(deltaIteration)
			iterations += 1

			if delta <= self.threshold:
				return V_s, deltaAll, iterations

	def value_iteration(self, gamma):
		V_s = np.zeros((self.size, self.size))
		iterations = 0
		dictMoves = {}
		delta_total = []
		for i in self.states:
			dictMoves[i] = 0

		while True:
			delta = 0
			delta_valueit = []
		#	Vcopy=np.copy(V_s)
			for s in self.states:  # s is an integer
				state = np.zeros((16)).astype(bool)  # boolean array
				state[s] = True
				all_actions_for_s = self.calc_action_value(state, V_s, gamma)
				best_action = np.max(all_actions_for_s)
				index_action = list(all_actions_for_s).index(best_action)
				action_letter = self.map_actions[index_action]
				# record best action for a certain state
				dictMoves[s] = action_letter

				# update Vs
			#	delta = max(delta, np.abs(V_s[np.reshape(state, (4, 4))]-Vcopy[np.reshape(state, (4, 4))]))
				delta = max(delta, np.abs(V_s[np.reshape(state, (4, 4))] - best_action))
				delta_valueit.append(np.abs(V_s[np.reshape(state, (4, 4))] - best_action))
				V_s[np.reshape(state, (4, 4))] = best_action
			delta_total.append(delta_valueit)
			iterations += 1

			if delta <= self.threshold:
				return V_s, dictMoves, delta_total, iterations

	def policy_iteration(self, gamma, method="howard"):
		# choose random policy to start with
		policy = np.ones([self.observation_space_n, self.size]) / 5
		iterationCount = 0
		count = 0
		dictPolicy = {}
		for i in range(16):
			dictPolicy[i] = 0

		while True:
			# evaluate the policy
			Vs, deltas, iterationsExecuted = self.evaluate_policy(policy, gamma)

			# Implementing simple policy iteration
			if method not in ["howard", "simple"]:
				raise ValueError("Method should be 'howard' or 'simple'")
			elif method == "howard":
				states = self.states
			elif method == "simple":
				states = self.states
				#states = [np.random.choice(self.states)]

			# policy improvement
			policy_stable = True
			for s in states:  # s is an integer
				# select best action under the current policy
				best_action_policy = np.argmax(policy[s])

				# save all possible action values
				s_all_actions_values = np.zeros(self.action_space_n)
				state = np.zeros((16)).astype(bool)  # boolean array
				state[s] = True
				for act, act_probability in enumerate(
						policy[s]):  # select all possible actions in this state (act is an integer)
					s_prime, reward, hit_grid, action_probability = self.sim_step_normal(state,
																						 act)  # pass a boolean array and an int
					s_prime_slip, reward_slip, hit_grid_slip, action_probability_slip = self.sim_step_slide(state, act)

					if hit_grid:
						s_all_actions_values[act] += reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]])
					else:
						s_all_actions_values[act] += action_probability * (
									reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]]))
						s_all_actions_values[act] += action_probability_slip * (
									reward_slip + (gamma * Vs[s_prime_slip[0][0], s_prime_slip[0][1]]))

				# the found best action based on the policy
				best_action_found = np.argmax(s_all_actions_values)  # returns the index of the highest value

				if best_action_policy != best_action_found:
					policy_stable = False
					policy[s] = np.eye(self.action_space_n)[
						best_action_found]  # maximaizes the probability of taking the action (Sets to 1) in the given state
					if method == "simple":
						break
				else:
					policy[s] = np.eye(self.action_space_n)[best_action_found]

			iterationCount += 1

			if policy_stable:
				for i in policy:
					n = 0
					for k in i:
						if k == 1:
							dictPolicy[count] = self.map_actions[n]
						else:
							n += 1
					count += 1

				return dictPolicy, Vs, iterationCount

	def simple_policy(self, gamma):
		# choose random policy to start with
		policy = np.ones([self.observation_space_n, self.size]) / 5
		iterationCount = 0
		count = 0
		dictPolicy = {}
		for i in range(16):
			dictPolicy[i] = 0

		while True:
			# evaluate the policy
			Vs, deltas, iterationsExecuted = self.evaluate_policy(policy, gamma)
			# Find improvable states
			improvable = []
			# Go through all non-terminal states
			for i_ in range(len(self.states)):
				# If none of the action-values within the states is 1 it is still improvable
				if np.all(np.array(self.states[i_]) != 1):
					# Save indices of improvable states
					improvable.append(i_)
			print("IMPRO:", improvable)
			states = [np.random.choice(np.array(self.states)[
				                           improvable])]  # [np.random.choice(self.states[np.array(policy) == "0"])] # only IMPROVABLE STATES
			print("STATES", states)
			# policy improvement
			policy_stable = False

			for s in states:  # s is an integer
				# select best action under the current policy
				best_action_policy = np.argmax(policy[s])

				# save all possible action values
				s_all_actions_values = np.zeros(self.action_space_n)
				state = np.zeros((16)).astype(bool)  # boolean array
				state[s] = True
				for act, act_probability in enumerate(
						policy[s]):  # select all possible actions in this state (act is an integer)
					s_prime, reward, hit_grid, action_probability = self.sim_step_normal(state,
					                                                                     act)  # pass a boolean array and an int
					s_prime_slip, reward_slip, hit_grid_slip, action_probability_slip = self.sim_step_slide(state,
					                                                                                        act)

					if hit_grid:
						s_all_actions_values[act] += reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]])
						#self.plot_data["simple"]["reward"].append(reward)
					else:
						s_all_actions_values[act] += action_probability * (
								reward + (gamma * Vs[s_prime[0][0], s_prime[0][1]]))
						s_all_actions_values[act] += action_probability_slip * (
								reward_slip + (gamma * Vs[s_prime_slip[0][0], s_prime_slip[0][1]]))
						#self.plot_data["simple"]["reward"].append(reward_slip)

				# the found best action based on the policy
				best_action_found = np.argmax(s_all_actions_values)  # returns the index of the highest value

				# CHANGE:Only stable if best action is not 0
				for val in s_all_actions_values:
					if val == 0:
						policy_stable = False

				# CHANGE: adjusted condition for simple policy iteration
				if best_action_found == best_action_policy:
					policy_stable = True

				if best_action_policy != best_action_found:
					policy_stable = False
					policy[s] = np.eye(self.action_space_n)[
						best_action_found]  # maximizes the probability of taking the action (Sets to 1) in the given state

				else:
					policy[s] = np.eye(self.action_space_n)[best_action_found]

			iterationCount += 1

			if policy_stable:
				for x, i in enumerate(policy):
					#print(i.max())
					#print(np.where(i == i.max())[0][0])
					#dictPolicy[x] = self.map_actions[np.where(i == i.max())[0][0]]

					n = 0
					for k in i:
						if k == 1:
							dictPolicy[count] = self.map_actions[n]
						else:
							n += 1
					count += 1

				return dictPolicy, Vs, iterationCount

	def to_coords(self, onehot_state):
		rem = 0
		while state % 3 != 0:
			state = state - 1
			rem += 1
		row = state // 3
		col = rem
		return (col, row)

	def get_state(self):
		"""Outputs state as integer
		from self.pos_mtx (mapping between lake and states)"""
		state_coords = np.where(self.pos_mtx == True)
		next_state = (state_coords[0] * self.size) + state_coords[1]
		return next_state

	def render(self):
		# show something, somewhere. Ideally on a screen.
		# We actually render this not useful by using the data model "__repr__",
		# but let's keep it for now.
		disp = self.lake_map.copy()
		disp[self.pos_mtx] = "#"
		print(disp)
		pass


def save_ts_pickle(filepath, var):
	import pickle
	import datetime as dt

	filepath = filepath + "_" + str(dt.datetime.today()).split('.')[0]
	with open(filepath, 'wb') as f:
		pickle.dump(var, f)
	print("saved to: " + filepath)
	pass


if __name__ == "__main__":
	env = Environment()
	import ipdb;

#ipdb.set_trace()
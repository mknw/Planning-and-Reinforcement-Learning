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
	
	def __init__(self, size=4, start_coords=(3, 0), wreck_coords=(2,2), goal_coords=(0,3),
				crack_coords=[(1,1), (1,3), (2,3), (3,1), (3,2), (3,3)]):
		self.size = size
		
		self.start_coords = start_coords
		self.wreck_coords = wreck_coords
		self.goal_coords = goal_coords
		self.crack_coords = crack_coords

		self.lake_map = np.zeros((size, size)).astype(str)
		self.state_space = ["S", "F", "C", "W", "G"] # start, frozen, crack, wreck, goal
		self.action_space = ['U', 'D', 'R', 'L'] # up, down, right, left
		self.reset()
		self.observation_space = self.lake_map.flatten()
		self.observation_space_n = self.lake_map.size
		self.action_space_n = len(self.action_space)

		self.map_actions = {k:v for k, v in enumerate(self.action_space)}
		pass

	def __repr__(self):
		# display frozen lake_map on terminal.
		return str(self.lake_map)


	def sample_action(self):
		# choose random action from self, action_space
		# (randomly called against epsylon)
		# return: action
		return np.random.choice(self.action_space)


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
		self.get_pos(reset=True) # set position matrix at "S"tart.
		pass

	def get_pos(self, reset=False):
		# get agent starting position
		if reset == True:
			self.pos = list(np.where(self.lake_map == "S"))
		# keep track of agent position by boolean pointer. 
		self.pos_mtx = np.zeros((self.size, self.size)).astype(bool)
		self.pos_mtx[tuple(self.pos)] = True
		print("Agent at position: " + str(self.pos))
		return self.pos_mtx


	def move(self, action_n):
		
		stop = False
		
		action = self.map_actions[action_n]
		prev_pos = np.copy(self.pos)
		if action == "U":
			self.pos[0] = np.max([0, self.pos[0]-1])
		elif action == "D":
			self.pos[0] = np.min([self.pos[0]+1, self.size-1])
		elif action == "L":
			self.pos[1] = np.max([0, self.pos[1] - 1])
		elif action == "R":
			self.pos[1] = np.min([self.pos[1] + 1, self.size - 1])
		else:
			raise ValueError("Possible action values are: " + str(self.action_space))
		
		# if no progres was made in the past movement, indicates agent 
		# is positioned along the maps' boundaries (STOP)
		if np.all(prev_pos == self.pos):
			stop = True
			current_state = self.lake_map[self.pos_mtx][0]
			return current_state, stop

		pos_mtx = self.get_pos()
		current_state = self.lake_map[self.pos_mtx][0]
		return current_state, stop

	def step(self, action):
		""" when step is performed, takes action from agent
		Returns:
		- destination_state,
		- reward,
		- "done" state (if goal achieved) """
		# Perform action, update position:
		current_state , _= self.move(action)

		if current_state == "C": # Falling?
			print("GAME OVER")
			done = True
			reward = -10
			return current_state, reward, done

		elif current_state == "F": # Slipping on frozen ice?
			if random.uniform(0, 1) <= .05:
				stop = False
				while not stop:
					next_state, stop = self.move(action)
					if next_state == "C":
						print("GAME OVER")
						reward = -10
						done = True
			else:
				reward = 0
				done = False

		elif next_state == "G": # Goal reached?
			print("YOU WON!")
			done = True # episode ended.
			reward = 100

		elif next_state == "W": # Wreck found?
			reward = 20
			done = False
			print("Wreck found!")
		
		self.render()
		state_coords = np.where(self.pos_mtx == True)
		next_state = (state_coords[0] * self.size) + state_coords[1]
		return next_state, reward, done


	def render(self):
		# show something, somewhere. Ideally on a screen.
		# We actually render this not useful by using the data model "__repr__",
		# but let's keep it for now. 
		disp = self.lake_map.copy()
		disp[self.pos_mtx] = "#"
		print(disp)
		pass


if __name__ == "__main__":
	env = Environment()
	import ipdb; ipdb.set_trace()


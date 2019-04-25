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
		# print("Environment renewed.")
		# generates self.pos_mtx to keep track of agent.
		self.get_pos(reset=True) # set position matrix at "S"tart.

		# flag for computing different probabilities in Howard PI
		self.movement_out_grid = False
		pass

	def get_pos(self, reset=False):
		""" Updates pos_mtx as boolean 'mask' for the lake_map.
		It does so by using self.pos coordinates, updated in the 
		'move' method."""

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
		
		self.movement_out_grid = False
		
		action = self.map_actions[action_n] #convert the integer to a letter

		#at the present moment pos_mtx actually contains the s (current) location
		#extract it
		self.pos = np.argwhere(self.pos_mtx==True)
		pos_backup = np.copy(self.pos)
		#prev_pos = np.copy(self.pos)

		#converted action movements to coordinates as pos now is a coordinate
		#updates pos to the s_prime coordinate based on the action executed
		if action == "U": # [-1,0]
			self.pos =  np.array(self.pos) + [-1,0]
		elif action == "D": #[1,0]
			self.pos = np.array(self.pos) + [1,0]
		elif action == "L": #[0,-1]
			self.pos = np.array(self.pos) + [0,-1]
		elif action == "R": #[0,1]
			self.pos = np.array(self.pos)+ [0,1]
		else:
			raise ValueError("Possible action values are: " + str(self.map_actions))

		#check if the agent moved oustide the grid
		#if true then adjust the value to the saved starting position (didnt move)
		if -1 in self.pos or 4 in self.pos:
			self.pos = pos_backup
			self.movement_out_grid=True
			self.slipping = False

		#if np.all(prev_pos == self.pos):
		#	stop = True
		y, x = zip(*self.pos)
		self.pos = np.array([[y],[x]])
		self.get_pos() # updates map cursor
		s_prime = self.lake_map[self.pos_mtx]#where s_prime is a letter from the map
		# create state vector by flattening state map.
		return s_prime

	def step(self, action, show_map_pos=False):
		"""
		When movement is performed, do the following:
		 - computes outcome (slipping, wreck, game over)
		 - assign 'done' and 'reward' for current step.
		Args:
		 - action (type: int, range: 0-3 inc.)

		Returns:
		 - destination_state (type: int, range: 0-15 inc.);
		 - reward (type: int);
		 - "done" (end game).
		"""
		# Perform action, update position:
		current_state = self.move(action) # move.
		self.movement_out_grid = False
		
		if self.movement_out_grid: # did the agent move at all from his starting pos?
			reward = 0
			done = False
		elif current_state == "C":  # Did he move onto a crack?
			# print("GAME OVER")
			done = True
			reward = -10
			next_state = self.get_state_int()
			return next_state, reward, done # return here to stepside "self.render()" after agent loss.
		elif current_state == "F" or current_state == "S":  # After moving, his he slipping on frozen ice?
			if random.uniform(0, 1) <= .05:
				self.slipping = True
				while self.slipping:
					next_state = self.move(action) # turns on self.slipping movement_out_grid when agent hits grid
					if next_state == "C":
						# print("GAME OVER")
						reward = -10
						done = True
					else:
						# print("slipping away...")
						reward = 0
						done = False
			else:
				reward = 0
				done = False
		elif current_state == "W":  # Wreck found?
			reward = 20
			done = False
			#print("Wreck found!")
		elif current_state == "G":  # Goal reached?
			#print("YOU WON!")
			done = True  # episode ended.
			reward = 100
		
		if show_map_pos:
			self.render()
		next_state = self.get_state_int()
		return next_state, reward, done

	def sim_step(self, state, action):
		""" 
		1. Update internal state (self.pos and self.pos_mtx)
		2. Simulates step for Howard PI, 
		Args:
			- 1hot vector, (type:array, 15values long), indicating
			  an arbitrary starting position, 
			- action (type int: 0-3)
		Returns:
			--- destination state, reward, done.
		""" 
		self.movement_out_grid = False
		self.pos_mtx = np.reshape(state, (4, 4))
		self.pos = list(np.where(self.pos_mtx == True))
		next_state, reward, done = self.step(action)
		return next_state, reward, done


	def to_coords(self, onehot_state):
		rem = 0
		while state % 3 !=0:
			state = state - 1
			rem += 1
		row = state // 3
		col = rem
		return (col, row)

	def get_state_int(self):
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

    ipdb.set_trace()

import numpy as np
import curses

""" rewards memo """
# ship wreck -> +20
# goal -> +100
# crack -> -10 and end episode.
"""" probs memo """
# moving: 0.95
# slipping: 0.05


class Environment(object):
	def __init__(self, start_coords=(3, 0), wreck_coords=(2,2), goal_coords=(0,3),
				crack_coords=[(1,1), (1,3), (2,3), (3,1), (3,2), (3,3)]):
		
		self.lake = np.zeros((4, 4)).astype(str)
		self.state_space = ["S", "F", "H", "W", "G"] # start, frozen, crack, wreck, goal
		self.action_space = ['U', 'D', 'R', 'L'] # up, down, right, left
		self.reset(start_coords, wreck_coords, goal_coords, crack_coords)
		# self.rewards # 100, 20, -10
		pass

	def __repr__(self):
		# display frozen lake on terminal.
		return str(self.lake)


	def sample_action(self):
		# choose random action from self, action_space
		# (randomly called against epsylon)
		# return: action
		return np.random.choice(self.action_space)


	def reset(self, start_coords, wreck_coords, goal_coords, crack_coords):

		# reset enviroment to pristine conditions.
		self.lake[:] = "F"
		self.lake[start_coords] = "S"
		self.lake[wreck_coords] = "W"
		self.lake[goal_coords] = "G"
		# crack coordinates are many. We 'unzip' the tuples and index the matrix only once:
		y_crack, x_crack = list(zip(*crack_coords))
		self.lake[y_crack, x_crack] = "C"
		pass

	def step(self, action):
		# when step is performed, takes action from agent
		# Returns:
		# destination_state,
		# reward,
		# "done" state (if goal achieved)
		pass

	def render(self):
		# show something, somewhere. Ideally on a screen.
		# We actually render this not useful by using the data model "__repr__",
		# but let's keep it for now. 
		print(self)
		pass


if __name__ == "__main__":
	env = Environment()
	import ipdb; ipdb.set_trace()


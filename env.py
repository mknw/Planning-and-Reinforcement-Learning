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
	def __init__(self, states, actions, rewards):
		self.state_space = states # F, H, S, 
		self.action_space = actions # up, down, left, right
		self.rewards = rewards # 100, 20, -10
		pass
	
	def sample_action(self):
		# choose random action from self, action_space
		# (randomly called against epsylon)
		# return: action
		return np.random.choice(self.action_space)


	def reset(self):
		# reset enviroment to pristine conditions.

	def step(self, action):
		# when step is performed, takes action from agent
		# Returns:
		# destination_state,
		# reward,
		# "done" state (if goal achieved)

	def render(self):
		# show something, somewhere. Ideally on a screen.





import numpy as np
import random


gamma = 0.9 # discounting rate
rewardSize_terminal = 100
reward_ship=20
reward_crack=-10
reward_normal=0
gridSize = 4
treasureState=[[2,2]]
terminationStates = [[1,1], [1,3], [2,3], [3,1], [3,2], [3,3],[0,3]]
crackStates = [[1,1], [1,3], [2,3], [3,1], [3,2], [3,3] ]
shipState=[[0,3]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 1000
action_probability=0.95
outside_grid_action_probability=1
inside_grid=True

#reward function
def actionRewardFunction(initialPosition, action):
	reward = reward_normal
	if initialPosition in terminationStates:
		if initialPosition in crackStates:
			return initialPosition, reward_crack
		if initialPosition in shipState:
			return initialPosition, rewardSize_terminal

	if initialPosition in treasureState:
		reward=reward_ship

	newState = np.array(initialPosition) + np.array(action)
	if -1 in newState or 4 in newState:
		newState = initialPosition
		inside_grid=False
	return newState, reward

#if __name__ == "__main__":

#set environment
V_s = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
	
#Policy evaluation (random policy) iterative
deltas = []
for i in range(numIterations):
	copyV_s = np.copy(V_s)
	#deltas will represent the amount of convergence achieved by each iteration
	deltaState = []
	for state in states:
		weightedRewards = 0
		for action in actions:
			s_prime, reward = actionRewardFunction(state, action)
			#random policy --> probability of taking an action 25% 1/len(Actions)
			if inside_grid:
				weightedRewards += (1/len(actions))*action_probability*(reward+(gamma*V_s[s_prime[0], s_prime[1]]))
			else:
				weightedRewards += (1 / len(actions))*outside_grid_action_probability*(reward + (gamma * V_s[s_prime[0], s_prime[1]]))
	
		deltaState.append(np.abs(copyV_s[state[0], state[1]]-weightedRewards))
		copyV_s[state[0], state[1]] = weightedRewards
	deltas.append(deltaState)
	V_s = copyV_s
	if i in [0,1,2,9, 99, numIterations-1]:
		print("Iteration {}".format(i+1))
		print(V_s)
		print("")

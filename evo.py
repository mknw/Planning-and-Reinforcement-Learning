import numpy as np
from env import Environment
from itertools import accumulate
from random import randint
import matplotlib.pyplot as plt


def evo():
	# Initializing number and size of populations, mutation probability and crossover factor
	# I chose to use single-point crossover
	population = 30
	generations = 10
	mutation = 0
	crossover = 0.6
	cross = "uniform"
	fitmode = "init"
	congraph = []
	np.random.seed(0)
	numit = 0
	# Dict for the population
	pop = {}
	pop[0] = []

	def msg(m):
		print("-----------------------------------------\n",
		      m,
		      "\n-----------------------------------------")


	E = Environment()
	msg("Evolutionary algorithm with "+str(generations)+" generations")
	# Create initial random population
	for individual in range(population):

		# Create random policy
		pol = np.random.random([E.observation_space_n, E.size])
		policy = np.ones([E.observation_space_n, E.size])

		# Normalize, so that each column sums up to 1 (saving in another array to not interfere with normalization)
		for i, row in enumerate(pol):
			for i_, entry in enumerate(row):
				policy[i][i_] = entry / sum(row)

		# It seems that due to the precision of the float the numbers still don't add up to 1 so we add the difference
		# between 1 and the sum of the row to a random number in the row until the criterion sum(row) == 1 is fulfilled

		for i, row in enumerate(policy):

			while sum(row) != 1.0:
				rest = 1 - sum(row)
				index = np.random.randint(len(row))
				policy[i][index] = policy[i][index] + rest

		pop[0].append(policy)

	# Loop through all generations
	for generation in range(1,generations-1):
	#fitness = [0]
	#generation = 0
	#while max(fitness) < 100:
		msg("Generation" + str(generation))
		pop[generation] = []
		fitness = []


		# Get fitness-score (maximum of state values)
		i = 0
		for individual in pop[generation-1]:
			numit += 1
			if fitmode == "max":
				fitness.append(max(E.evaluate_policy(individual,0.9)[0]))
			if fitmode == "init":
				fitness.append(E.evaluate_policy(individual,0.9)[0][0][2])
			i += 1
		print("max Fitness: ",max(fitness))

		# Sort population by fitness (descending)
		parents_sorted = np.array([x for _, x in sorted(zip(fitness, pop[generation - 1]), key=lambda pair: pair[0], reverse=True)])


		# Normalize fitness
		norm_fitness = [i/sum(fitness) for i in sorted(fitness, reverse=True)]
		while sum(norm_fitness) != 1.0:
			rest = 1 - sum(norm_fitness)
			index = np.random.randint(len(norm_fitness))
			norm_fitness[index] = norm_fitness[index] + rest

		# Get the accumulated normalized fitness
		acc_fitness = np.array(list(accumulate(norm_fitness)))

		# Choose parents and crossover
		for individual in range(population):
			# Choose parents
			better = []
			while True not in better:
				random = np.random.random()
				better = np.greater(np.array(acc_fitness), random)
			parent1 = parents_sorted[better][0]
			f = acc_fitness[better][0]
			better = []
			while True not in better:
				random = np.random.random()
				better = np.greater(np.array(np.delete(acc_fitness, f, axis=0)), random)

			# Second parent is chosen, can't be the first parent
			parent2 = np.delete(parents_sorted,parent1, axis=0)[better][0]

			# Single point cross-over
			if cross == "single":
				cross_index = int(len(parent1)/2)
				child = np.append(parent1[:cross_index],parent2[cross_index:], axis=0)
			# Uniform crossover
			if cross == "uniform":
				parents = [parent1,parent2]
				child = []
				for i in range(len(parent1)):
					child.append(parents[randint(0, 1)][i])
				child = np.array(child)
			# Mutation
			for i, policy in enumerate(child):
				random = np.random.random()
				if random <= mutation:
					random = np.random.random()
					index = randint(0,3)
					index2 = randint(0,3)
					temp = child[i][index]
					child[i][index] = child[i][index2]
					child[i][index2] = temp

			pop[generation].append(child)
			congraph.append(max(fitness))
			# Implementing elitism
			pop[generation].append(parents_sorted[0])

	plt.plot(range(len(congraph)),congraph)
	plt.savefig("evo_conv.png",dpi=300)
	plt.title("Convergence of the evolutionary algorithm")
	plt.ylim(100)
	plt.close()
	return E.evaluate_policy(parents_sorted[0],0.9)[0], numit
""" Group 2, Planning & Reinforcement Learning 2019
    Michael Accetto, Phillip Kersten
    Assignment Part 2"""



import numpy as np
import random
import pandas as pd
import env
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from learning import learning # sorry about that



if __name__ == "__main__":

	# can take 'Q', 'Q-ER', 'Q-ET' or 'SARSA'
	#methods = ["Q"]
	methods = ["Q-ER"]

	masterplot = []
	all_all_plots = []
	for method in methods:
		q_learning = learning(method)
		all_plots=[]

		# Record 100
		for i in range(10):
			plot_data = q_learning(alpha = .1, gamma = .6, epsilon = .1) # here alpha gamma and epsilon can be overwritten
			all_plots.append(plot_data)


		all_plots = np.array(all_plots)
		all_all_plots.append(all_plots)
		mean_plots = all_plots.mean(axis=0)
		sd_plots = all_plots.mean(axis=0)
		masterplot.append(mean_plots)
		df = pd.DataFrame(mean_plots)

		large = 22
		med = 16
		small = 12
		params = {'axes.titlesize': large,
		          'legend.fontsize': med,
		          'figure.figsize': (16, 10),
		          'axes.labelsize': med,
		          'axes.titlesize': med,
		          'xtick.labelsize': med,
		          'ytick.labelsize': med,
		          'figure.titlesize': large}
		plt.rcParams.update(params)
		plt.style.use('seaborn-whitegrid')
		sns.set_style("white")

		plt.figure(figsize=(16, 10), dpi=80)
		plt.plot(x=range(len(mean_plots)), y=mean_plots, color='tab:red')

		# Decoration
		plt.ylim(0, 150)

		xtick_labels = [str(x) for x in range(0,301,50)]
		xtick_location = [x for x in range(0,301,50)]

		plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center',
		           alpha=.7)
		plt.yticks(fontsize=12, alpha=.7)
		plt.title(method, fontsize=22)
		plt.grid(axis='both', alpha=.3)

		# Remove borders
		plt.gca().spines["top"].set_alpha(0.0)
		plt.gca().spines["bottom"].set_alpha(0.3)
		plt.gca().spines["right"].set_alpha(0.0)
		plt.gca().spines["left"].set_alpha(0.3)
		plt.show()

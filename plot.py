import matplotlib.pyplot as plt
import seaborn as sns


def plot(mean_plots,method,episodes=1000):

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
	plt.plot(range(len(mean_plots)), mean_plots, color='tab:red')

	# Decoration
	plt.ylim(0, 150)

	xtick_labels = [str(x) for x in range(0, episodes+1, 50)]
	xtick_location = [x for x in range(0, episodes+1, 50)]

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
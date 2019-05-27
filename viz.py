import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def smooth(y, box_pts=5):
	box = np.ones(box_pts) / box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth


def stat_ts(ts, dim=3):
	"""
	takes timeseries of shape:
	runs * episodes_per_run * stats
	Where stats are: [episode, epochs, penalties, tot_reward]
	"""
	n_runs = ts.shape[0]
	dim_series = ts[:,:,dim]
	
	# std error
	# dim_std = np.std(dim_series, axis=0)
	# stderr = dim_std / n_runs
	# average
	ts_averaged = np.sum(dim_series, axis=0) / n_runs
	return ts_averaged



def plot_ts(ts_avg, ts_stderr=False):
	"""
	Plot average over runs, 
	and standard error when provided.
	"""
	
	x = [i for i in range(len(ts_avg))]
	plt.plot(x, smooth(ts_avg), 'k-')
	
	if ts_stderr:
		ub = ts_avg + ts_stderr # error upper bound
		lb = ts_avg - ts_stderr # error lower bound
		plt.fill_between(x, lb, ub)
	plt.show()


def plot(mean_plots,method, fname, method_pars,epsi=0.1,episodes=1000,save=False):

	mean_plots = smooth(mean_plots, 20)
	mean_plots = mean_plots[:-20]
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

	plt.figure(figsize=(16, 10), dpi=300)
	plt.plot(range(len(mean_plots)), mean_plots, color='tab:red')

	# Decoration

	plt.ylim(0, 150)
	plt.xlim(0, 1000)

	xtick_labels = [str(x) for x in range(0, episodes+1, 50)]
	xtick_location = [x for x in range(0, episodes+1, 50)]

	plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, horizontalalignment='center',
	           alpha=.7)
	plt.yticks(fontsize=12, alpha=.7)
	plt.title(method+"\n" + method_pars, fontsize=20)
	plt.grid(axis='both', alpha=.3)

	# Remove borders
	plt.gca().spines["top"].set_alpha(0.0)
	plt.gca().spines["bottom"].set_alpha(0.3)
	plt.gca().spines["right"].set_alpha(0.0)
	plt.gca().spines["left"].set_alpha(0.3)
	if save:
		plt.savefig(fname, dpi=300)
		plt.close()

	# plt.show()

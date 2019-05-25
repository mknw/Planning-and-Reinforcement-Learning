import numpy as np
from matplotlib import pyplot as plt


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
	dim_std = np.std(dim_series, axis=0)
	stderr = dim_std / n_runs
	# average
	ts_averaged = np.sum(dim_series, axis=0) / n_runs
	return ts_averaged, stderr



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

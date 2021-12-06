import pickle 
import matplotlib.pyplot as plt
import numpy as np 
from sys import argv
import sys
import os 
import argparse
import json
import pandas as pd 
import matplotlib.ticker as mticker
from scipy.stats import ttest_ind
from scipy.stats import median_test
def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--root-dir', type=str, default='log')
    #parser.add_argument('--render', type=float, default=False)
    parser.add_argument('--value-type', type=str, default='average')
    parser.add_argument('--baseline-dir', type=str, default='offpolicy_benchmark/')
    parser.add_argument('--baseline-algo', type=str, default='sac')
    parser.add_argument('--save-fig', dest='save_fig', action='store_true')
    parser.add_argument('--compute', dest='compute', action='store_true')
    return parser.parse_args()

def parse_pkl_file(path_to_seed, interpolated_length, max_steps=1e6):
	x_new = np.arange(0, max_steps, interpolated_length)
	path_to_pkl = os.path.join(path_to_seed, 'log.pkl')
	with open(path_to_pkl, 'rb') as f:
		data = pickle.load(f)
		timesteps = np.array(data['total_steps'])
		score = np.array(data['mu_score'])
		interpolated_score = np.interp(x_new, timesteps, score).reshape(1,x_new.shape[0])
	return interpolated_score

def parse_csv_file(path_to_seed, interpolated_length, max_steps=1e6):
	x_new = np.arange(0, max_steps, interpolated_length)
	path_to_csv = os.path.join(path_to_seed, 'test_rew.csv')
	data = pd.read_csv(path_to_csv)
	
	data = data[data['env_step'] <= max_steps]
	score = np.array(data['rew'])
	timesteps = np.array(data['env_step'])
	interpolated_score = np.interp(x_new, timesteps, score).reshape(1,x_new.shape[0])

	return interpolated_score 

def parse_multiple_seed(parent_folder, interpolated_length=10000, max_steps=1e6, type='pkl'):
	seed_folders = next(os.walk(parent_folder))[1]
	result_multiple_seeds = np.zeros((len(seed_folders), int(max_steps//interpolated_length)))

	for i, folder in enumerate(seed_folders):
		path = os.path.join(parent_folder, folder)
		if type == 'pkl':
			result_multiple_seeds[i, :] = parse_pkl_file(path, interpolated_length)
		elif type == 'csv':
			result_multiple_seeds[i, :] = parse_csv_file(path, interpolated_length)
		else:
			raise Exception('Not implemented')

	return result_multiple_seeds

color_ind = 0
COLORS = ([
    # deepmind style
    '#0072B2',
    '#009E73',
    '#D55E00',
    '#CC79A7',
    # '#F0E442',
    '#d73027',  # RED
    # built-in color
    'blue', 'red', 'pink', 'cyan', 'magenta', 'yellow', 'black', 'purple',
    'brown', 'orange', 'teal', 'lightblue', 'lime', 'lavender', 'turquoise',
    'darkgreen', 'tan', 'salmon', 'gold', 'darkred', 'darkblue', 'green',
    # personal color
    '#313695',  # DARK BLUE
    '#74add1',  # LIGHT BLUE
    '#f46d43',  # ORANGE
    '#4daf4a',  # GREEN
    '#984ea3',  # PURPLE
    '#f781bf',  # PINK
    '#ffc832',  # YELLOW
    '#000000',  # BLACK
])

#plt.style.use('seaborn')
args = get_args()

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)

shape = (2, 6)
colspan = 2
rowspan = 1
ax1 = plt.subplot2grid(shape=shape, loc=(0,0), colspan=colspan, rowspan=rowspan)
ax2 = plt.subplot2grid(shape, (0,2), colspan=colspan, rowspan=rowspan)
ax3 = plt.subplot2grid(shape, (0,4), colspan=colspan, rowspan=rowspan)
ax4 = plt.subplot2grid(shape, (1,1), colspan=colspan, rowspan=rowspan)
ax5 = plt.subplot2grid(shape, (1,3), colspan=colspan, rowspan=rowspan)

axes = [ax1, ax2, ax3, ax4, ax5]

list_of_env = ['Walker2d-v3', 'Hopper-v3', 'Ant-v3', 'Humanoid-v3', 'HalfCheetah-v3']

x_new = np.arange(0, 1e6, 10000)
x_web = np.arange(0, 1e6, 5000)
for i, env in enumerate(list_of_env):
	etd_result = parse_multiple_seed(f"ETD3/{env}", type='csv')
	cemtd_result = parse_multiple_seed(f"CEM_TD3/{env}", type='pkl')
	baseline_result = parse_multiple_seed(f"{args.baseline_dir}/{env}/{args.baseline_algo}", interpolated_length=5000, type='csv')

	if args.compute:
		print('\nenv:', env)
		print('ETD3', round(np.median(etd_result[:,-1]),3), '&', round(np.mean(etd_result[:,-1]),3), '&',round(np.std(etd_result[:,-1]),3))
		print('CEM-TD3', round(np.median(cemtd_result[:,-1]),3), '&', round(np.mean(cemtd_result[:,-1]),3), '&',round(np.std(cemtd_result[:,-1]),3))
		print(f'{args.baseline_algo}', round(np.median(baseline_result[:,-1]),3), '&', round(np.mean(baseline_result[:,-1]),3), '&',round(np.std(baseline_result[:,-1]),3))
		
		_, m1, _, _ = median_test(etd_result[:,-1], baseline_result[:,-1], ties='above')
		_, m2, _, _ = median_test(etd_result[:,-1], cemtd_result[:,-1], ties='above')
		_, t1 = ttest_ind(etd_result[:,-1], baseline_result[:,-1])
		_, t2 = ttest_ind(etd_result[:,-1], cemtd_result[:,-1])

		print('ttest(etd-cemtd) & mtest(etd-cemtd) & ttest(etd-td3) & mtest(etd-td3)')
		print(f'{round(t2,5)} & {round(m2,5)} & {round(t1,5)} & {round(m1,5)}')

	if args.value_type == 'average':
		etd = np.mean(etd_result, axis=0)
		etd_std = np.std(etd_result, axis=0)
		cemtd = np.mean(cemtd_result, axis=0)
		cemtd_std = np.std(cemtd_result, axis=0)
		baseline = np.mean(baseline_result, axis=0)
		baseline_std = np.std(baseline_result, axis=0)
		axes[i].fill_between(x_new, etd + etd_std, etd - etd_std, alpha=0.3, color=COLORS[0])
		if env != 'Humanoid-v3':
			axes[i].fill_between(x_new, cemtd + cemtd_std, cemtd - cemtd_std, alpha=0.2, color=COLORS[1])
		axes[i].fill_between(x_web, baseline + baseline_std, baseline - baseline_std, alpha=0.2, color=COLORS[2])

	elif args.value_type == 'median':
		etd = np.median(etd_result, axis=0)
		q75, q25 = np.percentile(etd_result, [90 ,10], axis=0)
		axes[i].fill_between(x_new, q75, q25, alpha=0.3, color=COLORS[0])

		cemtd = np.median(cemtd_result, axis=0)
		q75, q25 = np.percentile(cemtd_result, [90 ,10], axis=0)
		if env != 'Humanoid-v3':
			axes[i].fill_between(x_new, q75, q25, alpha=0.2, color=COLORS[1])
		
		baseline = np.median(baseline_result, axis=0)
		q75, q25 = np.percentile(baseline_result, [90 ,10], axis=0)
		axes[i].fill_between(x_web, q75, q25, alpha=0.2, color=COLORS[2])

	axes[i].plot(x_new, etd, label='ETD3', color=COLORS[0])
	if env != 'Humanoid-v3':
		axes[i].plot(x_new, cemtd, label='CEM-TD3', color=COLORS[1])
	axes[i].plot(x_web, baseline, label=f'{args.baseline_algo}', color=COLORS[2])
	axes[i].legend()
	axes[i].set_xlabel('env steps')
	if i==0:
		axes[i].set_ylabel('score')
	axes[i].set_title(f'{env}')
plt.tight_layout()

if args.save_fig:
	plt.savefig(f'./nics_{args.value_type}.pdf')
else:
	plt.show()
sys.exit()

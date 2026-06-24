import argparse
import pickle
import matplotlib.pyplot as plt
import h5py
from itertools import combinations

from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from plot.SR_result import get_GA_statistics
from experiment.formulas import get_ground_truth


FORMULAS = ["sho", "iig", "fsa", "tre"]

parser = argparse.ArgumentParser(description='Plot the model performance on the experiment formulas')

parser.add_argument('--path_template', default=EXPERIMENT_RESULT_PATH_TEMPLATE, 
	help='path template for the experiment results')

args = parser.parse_args()


results = [[np.array([x]) for x in pickle.load(open(args.path_template.format(i), 'rb'))] for i in FORMULAS]

for i in range(len(FORMULAS)):
	results[i].append(np.array([get_ground_truth(FORMULAS[i])]*len(results[i][0][0])))

score_means, score_stds, \
complexity_means, complexity_stds, \
match_means, match_stds, \
score_baseline_means, score_baseline_stds, \
complexity_baseline_means, complexity_baseline_stds, \
match_baseline_means, match_baseline_stds = get_GA_statistics(results)

score_means = np.array([x[0] for x in score_means])
score_stds = np.array([x[0] for x in score_stds])
score_baseline_means = np.array([x[0] for x in score_baseline_means])
score_baseline_stds = np.array([x[0] for x in score_baseline_stds])
score_improve = (score_means - score_baseline_means) / score_baseline_means

complexity_means = np.array([x[0] for x in complexity_means])
complexity_stds = np.array([x[0] for x in complexity_stds])
complexity_baseline_means = np.array([x[0] for x in complexity_baseline_means])
complexity_baseline_stds = np.array([x[0] for x in complexity_baseline_stds])
complexity_improve = -(complexity_means - complexity_baseline_means) / complexity_baseline_means

match_means = np.array([x[0] for x in match_means])
match_stds = np.array([x[0] for x in match_stds])
match_baseline_means = np.array([x[0] for x in match_baseline_means])
match_baseline_stds = np.array([x[0] for x in match_baseline_stds])
match_improve = (match_means - match_baseline_means) / match_baseline_means



plt.rc('font', size=10)
x = np.arange(len(FORMULAS)) 
width = 0.35

fig, ax = plt.subplots(1,3, figsize=(24,4))

rects1 = ax[0].bar(x - width/2, score_means, width, yerr=score_stds, label='DESTROI')
rects2 = ax[0].bar(x + width/2, score_baseline_means, width, yerr=score_baseline_stds, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[0].text(rects1[i].get_x() + width, height+0.015, '{:.0%}'.format(score_improve[i]), ha='center', va='bottom')
ax[0].set_ylabel(r'$R^2$ (higher is better)')
ax[0].set_xticks(x)
ax[0].set_xticklabels(FORMULAS)
ax[0].set_ylim(0.4, 1.05)
ax[0].legend()

rects1 = ax[1].bar(x - width/2, match_means, width, yerr=match_stds, label='DESTROI')
rects2 = ax[1].bar(x + width/2, match_baseline_means, width, yerr=match_baseline_stds, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[1].text(rects1[i].get_x() + width, height+0.015, '{:.0%}'.format(match_improve[i]), ha='center', va='bottom')
ax[1].set_ylabel(r'discovery rate (higher is better)')
ax[1].set_xticks(x)
ax[1].set_xticklabels(FORMULAS)
ax[1].set_ylim(0.0, 0.9)
ax[1].legend()

rects1 = ax[2].bar(x - width/2, complexity_means, width, yerr=complexity_stds, label='DESTROI')
rects2 = ax[2].bar(x + width/2, complexity_baseline_means, width, yerr=complexity_baseline_stds, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[2].text(rects1[i].get_x() + width, height+0.25, '{:.0%}'.format(complexity_improve[i]), ha='center', va='bottom')
ax[2].set_ylabel(r'# of internal nodes (lower is better)')
ax[2].set_xticks(x)
ax[2].set_xticklabels(FORMULAS)
ax[2].set_ylim(0, 65)
ax[2].legend()

plt.show()
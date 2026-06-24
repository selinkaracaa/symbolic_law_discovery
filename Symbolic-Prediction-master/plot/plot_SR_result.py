import argparse
import matplotlib.pyplot as plt
import pickle

from common.preset import *
from plot.SR_result import get_GA_statistics


NVARS = [2, 3, 5, 10]
NGEN = [4, 8, 12, 16, 20]

parser = argparse.ArgumentParser(description='Plot the model performance for genetic algorithm')

parser.add_argument('--path_template', default=GA_RESULT_PATH_TEMPLATE, 
	help='path template for the genetic algorithm results')

args = parser.parse_args()


results = [pickle.load(open(args.path_template.format(i), 'rb')) for i in NVARS]

score_means, score_stds, \
complexity_means, complexity_stds, \
match_means, match_stds, \
score_baseline_means, score_baseline_stds, \
complexity_baseline_means, complexity_baseline_stds, \
match_baseline_means, match_baseline_stds = get_GA_statistics(results)


plt.rc('font', size=10)
fig, axes = plt.subplots(3,4,figsize=(20,10))
for i, ax in enumerate(axes[0]):
    ax.errorbar(NGEN, score_means[i], yerr=score_stds[i], label='DESTROI')
    ax.errorbar(NGEN, score_baseline_means[i], yerr=score_baseline_stds[i], label='baseline')
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1)

for i, ax in enumerate(axes[1]):
    ax.errorbar(NGEN, match_means[i], yerr=match_stds[i], label='DESTROI')
    ax.errorbar(NGEN, match_baseline_means[i], yerr=match_baseline_stds[i], label='baseline')
    ax.legend(loc='lower right')
    ax.set_ylim(0.1, 0.6)

for i, ax in enumerate(axes[2]):
    ax.errorbar(NGEN, complexity_means[i], yerr=complexity_stds[i], label='DESTROI')
    ax.errorbar(NGEN, complexity_baseline_means[i], yerr=complexity_baseline_stds[i], label='baseline')
    ax.legend(loc='lower right')
    ax.set_ylim(2, 6)

for ax in axes.flatten():
    ax.set_xticks(NGEN)

for ax in axes[2]:
    ax.set_xlabel('generations')

for i, ax in enumerate(axes[0]):
	ax.set_title(r'$k=$'+str(NVARS[i]))

axes[0,0].set_ylabel(r'$R^2$')
axes[1,0].set_ylabel('discovery rate')
axes[2,0].set_ylabel('# of internal nodes')
axes[0,-1].yaxis.set_label_position("right")
axes[1,-1].yaxis.set_label_position("right")
axes[2,-1].yaxis.set_label_position("right")
axes[0,-1].set_ylabel('higher is better',rotation=-90, labelpad=25)
axes[1,-1].set_ylabel('higher is better',rotation=-90, labelpad=25)
axes[2,-1].set_ylabel('lower is better',rotation=-90, labelpad=25)

plt.show()
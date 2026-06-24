import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

from common.preset import *
from plot.SR_result import get_GA_statistics


NVARS = [2, 3, 5, 10]
NGEN = [4, 8, 12, 16, 20]

parser = argparse.ArgumentParser(description='Plot the model performance for genetic algorithm at specified generation')

parser.add_argument('--path_template', default=GA_RESULT_PATH_TEMPLATE, 
	help='path template for the genetic algorithm results')
parser.add_argument('--gen', default=NGEN[-1], 
	help='generation at which to evaluate the genetic algorithm results, available: {}'.format(NGEN))

args = parser.parse_args()


results = [pickle.load(open(args.path_template.format(i), 'rb')) for i in NVARS]

score_means, score_stds, \
complexity_means, complexity_stds, \
match_means, match_stds, \
score_baseline_means, score_baseline_stds, \
complexity_baseline_means, complexity_baseline_stds, \
match_baseline_means, match_baseline_stds = get_GA_statistics(results)

score_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in score_means])
score_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in score_stds])
score_baseline_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in score_baseline_means])
score_baseline_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in score_baseline_stds])
score_improve = (score_means_at_gen - score_baseline_means_at_gen) / score_baseline_means_at_gen

complexity_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in complexity_means])
complexity_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in complexity_stds])
complexity_baseline_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in complexity_baseline_means])
complexity_baseline_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in complexity_baseline_stds])
complexity_improve = -(complexity_means_at_gen - complexity_baseline_means_at_gen) / complexity_baseline_means_at_gen

match_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in match_means])
match_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in match_stds])
match_baseline_means_at_gen = np.array([x[NGEN.index(args.gen)] for x in match_baseline_means])
match_baseline_stds_at_gen = np.array([x[NGEN.index(args.gen)] for x in match_baseline_stds])
match_improve = (match_means_at_gen - match_baseline_means_at_gen) / match_baseline_means_at_gen



plt.rc('font', size=10)
x = np.arange(len(NVARS)) 
width = 0.35

fig, ax = plt.subplots(1,3, figsize=(24,4))

rects1 = ax[0].bar(x - width/2, score_means_at_gen, width, yerr=score_stds_at_gen, label='DESTROI')
rects2 = ax[0].bar(x + width/2, score_baseline_means_at_gen, width, yerr=score_baseline_stds_at_gen, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[0].text(rects1[i].get_x() + width, height+0.015, '{:.0%}'.format(score_improve[i]), ha='center', va='bottom')
ax[0].set_ylabel(r'$R^2$ (higher is better)')
ax[0].set_xlabel(r'$k$')
ax[0].set_xticks(x)
ax[0].set_xticklabels(NVARS)
ax[0].set_ylim(0.6, 0.92)
ax[0].legend()

rects1 = ax[1].bar(x - width/2, match_means_at_gen, width, yerr=match_stds_at_gen, label='DESTROI')
rects2 = ax[1].bar(x + width/2, match_baseline_means_at_gen, width, yerr=match_baseline_stds_at_gen, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[1].text(rects1[i].get_x() + width, height+0.015, '{:.0%}'.format(match_improve[i]), ha='center', va='bottom')
ax[1].set_ylabel(r'discovery rate (higher is better)')
ax[1].set_xlabel(r'$k$')
ax[1].set_xticks(x)
ax[1].set_xticklabels(NVARS)
ax[1].set_ylim(0.3, 0.56)
ax[1].legend()

rects1 = ax[2].bar(x - width/2, complexity_means_at_gen, width, yerr=complexity_stds_at_gen, label='DESTROI')
rects2 = ax[2].bar(x + width/2, complexity_baseline_means_at_gen, width, yerr=complexity_baseline_stds_at_gen, label='Baseline')
for i in range(4):
    height = max(rects1[i].get_height(), rects2[i].get_height())
    ax[2].text(rects1[i].get_x() + width, height+0.25, '{:.0%}'.format(complexity_improve[i]), ha='center', va='bottom')
ax[2].set_ylabel(r'# of internal nodes (lower is better)')
ax[2].set_xlabel(r'$k$')
ax[2].set_xticks(x)
ax[2].set_xticklabels(NVARS)
ax[2].set_ylim(3, 6.5)
ax[2].legend()

plt.show()
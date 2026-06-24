import numpy as np

from common.symbolic_tree import is_symbolic_form_match


def get_statistics(values):
	return np.mean(values, axis=1), np.std(values, axis=1) / np.sqrt(values.shape[1])


def complexity(programs):
	complexity = np.zeros((programs.shape[0], programs.shape[1]))
	for i in range(programs.shape[0]):
		for j in range(programs.shape[1]):
			complexity[i, j] = programs[i][j].count('(')
	return complexity


def is_match(program, ground_truth):
	return is_symbolic_form_match(program.replace(' ', ''), ground_truth)


def get_matches(programs, ground_truths):
	matches = np.zeros((len(programs), len(programs[0])))
	for i in range(programs.shape[0]):
		for j in range(programs.shape[1]):
			matches[i, j] = is_match(programs[i][j], ground_truths[j])
	return matches

def get_GA_statistics(results):
	nvars = len(results)
	ngens = len(results[0])

	score_means = np.zeros((nvars, ngens))
	score_stds = np.zeros((nvars, ngens))
	complexity_means = np.zeros((nvars, ngens))
	complexity_stds = np.zeros((nvars, ngens))
	match_means = np.zeros((nvars, ngens))
	match_stds = np.zeros((nvars, ngens))
	score_baseline_means = np.zeros((nvars, ngens))
	score_baseline_stds = np.zeros((nvars, ngens))
	complexity_baseline_means = np.zeros((nvars, ngens))
	complexity_baseline_stds = np.zeros((nvars, ngens))
	match_baseline_means = np.zeros((nvars, ngens))
	match_baseline_stds = np.zeros((nvars, ngens))

	for i in range(nvars):
		score_means[i], score_stds[i] = get_statistics(np.clip(results[i][2], 0, 1))
		score_baseline_means[i], score_baseline_stds[i] = get_statistics(np.clip(results[i][3], 0, 1))
		complexity_means[i], complexity_stds[i] = get_statistics(complexity(results[i][0]))
		complexity_baseline_means[i], complexity_baseline_stds[i] = get_statistics(complexity(results[i][1]))
		match_means[i], match_stds[i] = get_statistics(get_matches(results[i][0], results[i][4]))
		match_baseline_means[i], match_baseline_stds[i] = get_statistics(get_matches(results[i][1], results[i][4]))

	return score_means, score_stds, \
		complexity_means, complexity_stds, \
		match_means, match_stds, \
		score_baseline_means, score_baseline_stds, \
		complexity_baseline_means, complexity_baseline_stds, \
		match_baseline_means, match_baseline_stds
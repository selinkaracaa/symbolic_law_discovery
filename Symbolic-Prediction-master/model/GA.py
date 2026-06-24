from gplearn.genetic import SymbolicRegressor

from common.symbolic_tree import OPERATOR_LIST
from common.preset import *


def GA_Baseline():
	return SymbolicRegressor(
		function_set=tuple(OPERATOR_LIST), 
		init_depth=(MIN_DEPTH, MAX_DEPTH),
		generations=GA_BASELINE_GENERATION,
		population_size=GA_BASELINE_POPULATION,
		n_jobs=-1)

def GA(func_set, generation):
	return SymbolicRegressor(
		function_set=func_set, 
		init_depth=(MIN_DEPTH, MAX_DEPTH),
		generations=generation,
		population_size=GA_POPULATION,
		parsimony_coefficient=GA_PARSIMONY,
		n_jobs=-1)
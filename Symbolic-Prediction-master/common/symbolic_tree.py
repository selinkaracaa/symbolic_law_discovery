import numpy as np
import random


OPERATOR_LIST = ['add', 'mul', 'inv','sqrt', 'log', 'sin']
OPERATOR_ARGS = {'add':2, 'mul':2, 'inv':1,'sqrt':1, 'log':1, 'sin':1}


"""
A Node in the symbolic tree

- is_terminal: a terminal node corresponda to a constant or variable, an
	internal node corresponds to an operator
- variable: index of the variable, None if node is operator, -1 if node is a
	constant
- constant: constant value, None if node is operator or variable
- operator: index of the operator, None if node is terminal
"""
class Node:
	def __init__(self, is_terminal, variable, constant, operator):
		self.is_terminal = is_terminal
		self.variable = variable
		self.constant = constant
		self.operator = operator
		self.children = []


def get_random_constant_node(constant_range):
	return Node(True, -1, random.uniform(*constant_range), None)


def get_random_variable_node(variable_count):
	return Node(True, random.randrange(variable_count), None, None)


def get_random_terminal_node(
	variable_count,
	constant_prob, 
	constant_range,
	can_be_constant,
):
	if random.random() < constant_prob and can_be_constant:
		return get_random_constant_node(constant_range)
	else:
		return get_random_variable_node(variable_count)


def get_random_operator_node():
	return Node(False, None, None, random.randrange(len(OPERATOR_LIST)))


# randomly generate the children of the given node. The children can be forced
# to be termninal or non terminal. Otherwise, they are generated according to
# the terminal probability
def populate_children(
	node, 
	variable_count, 
	terminal_prob,
	constant_prob, 
	constant_range,
	force_terminal, 
	force_non_terminal,
):

	count = OPERATOR_ARGS[OPERATOR_LIST[node.operator]]

	for i in range(count):
		# a node cannot have all children to be constant
		all_constant = np.all(
			[child.constant is not None for child in node.children]
		)
		can_be_constant = i < count - 1 or not all_constant

		if force_terminal:
			child = get_random_terminal_node(
				variable_count, constant_prob, constant_range, can_be_constant
			)
		elif force_non_terminal:
			child = get_random_operator_node()
		elif random.random() < terminal_prob:
			child = get_random_terminal_node(
				variable_count, constant_prob, constant_range, can_be_constant
			)
		else:
			child = get_random_operator_node()

		node.children.append(child)


# recursively expand the given node to a fully grown symbolic tree satisfying
# the conditions.
def expand_node(
	node, 
	variable_count, 
	terminal_prob, 
	constant_prob, 
	constant_range,
	depth_range,
):

	if node.is_terminal:
		return

	min_depth, max_depth = depth_range
	populate_children(
		node, variable_count, terminal_prob, constant_prob, constant_range, 
		max_depth <= 0, min_depth > 0
	)

	for child in node.children:
		expand_node(
			child, variable_count, terminal_prob, constant_prob, constant_range,
			[min_depth-1, max_depth-1]
		)


def generate_random_tree(
	variable_count, 
	terminal_prob, 
	constant_prob, 
	constant_range,
	depth_range,
):
	min_depth, max_depth = depth_range
	root = get_random_operator_node()
	expand_node(
		root, variable_count, terminal_prob, constant_prob, constant_range, 
		[min_depth-1, max_depth-1]
	)
	return root


def get_depth(node):
	if node.is_terminal:
		return 0
	else:
		return max([get_depth(child) for child in node.children]) + 1


# return an array labeling the existence of the operators starting from the
# given node
def get_classification_labels(node):
	result = np.zeros(len(OPERATOR_LIST))
	if node.is_terminal:
		return result
	result[node.operator] = 1
	for child in node.children:
		result += get_classification_labels(child)
	return np.sign(result)


def get_max_var(node):
	if node.is_terminal:
		return max(node.variable, 0)
	return max([get_max_var(child) for child in node.children])


def is_symbolic_form_match(s1, s2):
	for x in OPERATOR_LIST:
		if s1.count(x) != s2.count(x):
			return False

	max_var = max(
		get_max_var(get_node_from_symbolic_form(s1)), 
		get_max_var(get_node_from_symbolic_form(s2))
	)

	for i in range(max_var):
		if s1.count('X'+str(i)) != s2.count('X'+str(i)):
			return False
	return True


# get the symbolic string from of the node
def get_symbolic_form(node):
	if node.is_terminal:
		if node.variable == -1:
			return str(node.constant)
		else :
			return "X" + str(node.variable)
	subforms = [get_symbolic_form(child) for child in node.children]
	return OPERATOR_LIST[node.operator] + "(" + ",".join(subforms) + ")"


# retrive the symbolic tree from the symbolic string form
def get_node_from_symbolic_form(s):
	i = s.find('(')
	
	if i == -1:
		if s[0] == 'X':
			# a single variable
			return Node(True, int(s[1:]), None, None)
		else:
			# a single constant
			return Node(True, -1, float(s), None)
	
	op = s[:i]
	arg = s[i+1:-1]
	
	node = Node(False, None, None, OPERATOR_LIST.index(op))

	children_s = []
	pcount = 0
	start = 0
	for j in range(len(arg)):
		if arg[j] == '(':
			pcount += 1
		if arg[j] == ')':
			pcount -= 1
		if arg[j] == ',' and pcount == 0:
			children_s.append(arg[start:j])
			start = j+1
	children_s.append(arg[start:])

	node.children = [
		get_node_from_symbolic_form(child_s) for child_s in children_s
	]
	
	return node

def get_value(node, variable_values, epsilon=1e-5):
	if node.is_terminal:
		if node.variable == -1:
			return node.constant
		else:
			return variable_values[node.variable]
	else:
		sub_values = [
			get_value(child, variable_values) for child in node.children
		]
		if OPERATOR_LIST[node.operator] == "add":
			return sub_values[0] + sub_values[1]
		if OPERATOR_LIST[node.operator] == "mul":
			return sub_values[0] * sub_values[1]
		if OPERATOR_LIST[node.operator] == "inv":
			direction = 1 if sub_values[0] >= 0 else -1
			return 1 / (sub_values[0] + direction * epsilon)
		if OPERATOR_LIST[node.operator] == "sqrt":
			return np.sqrt(np.abs(sub_values[0]))
		if OPERATOR_LIST[node.operator] == "log":
			return np.log(np.abs(sub_values[0]) + epsilon)
		if OPERATOR_LIST[node.operator] == "sin":
			return np.sin(sub_values[0])
	return 0
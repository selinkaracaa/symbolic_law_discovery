import argparse
import numpy as np
import tensorflow as tf

from common.symbolic_tree import OPERATOR_LIST
from common.preset import *
from common.write_h5_file import write_h5_file
from data_generation.generate_data import generate_projections
from decoder.run_decoder import predict_for_op, get_att_for_op
from encoder.MLP.run_encoder import encode as MLP_encode
from encoder.super_resolution.prepare_dataset import get_naive_encodings
from encoder.super_resolution.run_encoder import encode as SR_encoder
from experiment.formulas import get_formula
from model.CNN import CNN
from model.MIL_Attention import MIL_Attention
from model.super_resolution import EDSR


SEED = 1599281782


def generate_experiment_data(
	formula_name, nsample=EXPERIMENT_NSAMPLE, ndim=NDIM, noise=EXPERIMENT_NOISE,
	var_range_low=VAR_RANGE_LOW, var_range_high=VAR_RANGE_HIGH):

	formula, k = get_formula(formula_name)
	X = np.random.uniform(low=var_range_low, high=var_range_high, size=(nsample, k))
	Y = np.array([formula(*x) + np.random.normal(0, noise) for x in X])[:, None]
	inp = np.append(X, Y, axis=1)
	if k == NDIM:
		return inp, np.array([inp])
	else:
		return inp, generate_projections(np.array([inp]), ndim=ndim)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run experiment with specified formula')
	parser.add_argument('--formula', help='name of formula used')
	parser.add_argument('--encoder', default=ENCODING_SCHEME, help='encoding scheme used')
	parser.add_argument('--path', default='default', help='path to save the experiment data')
	parser.add_argument('--encoder_path_template', default=SUPERR_ENCODER_WEIGHT_PATH_TEMPLATE, 
		help='path template for the super-resolution encoder')
	parser.add_argument('--decoder_path_template', default=DECODER_WEIGHT_PATH_TEMPLATE, 
		help='path template for the decoder')
	parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
	args = parser.parse_args()

	if args.path == 'default':
		args.path = EXPERIMENT_PATH_TEMPLATE.format(args.formula)
		
	np.random.seed(SEED)

	inp, inp_proj = generate_experiment_data(args.formula)

	projected = len(inp_proj.shape) == 4
	nvar = inp.shape[-1] - 1

	if args.encoder == 'MLP':
		X = MLP_encode(inp_proj)
	elif args.encoder == 'super resolution':
		encoder = EDSR()
		encoder.load_weights(args.encoder_path_template.format(nvar))
		lr = get_naive_encodings(inp_proj)
		X = SR_encoder(encoder, lr, batch_size=args.batch_size)

	if projected:
		decoder = MIL_Attention()
	else:
		decoder = CNN()

	pred = np.zeros(len(OPERATOR_LIST))

	for i in range(len(OPERATOR_LIST)):
		pred[i] = predict_for_op(
			decoder, X, i, nvar=nvar, 
			path_template=args.decoder_path_template
		)

	print('Data saved to:', args.path)
	write_h5_file(args.path, {
		'inp':inp,
		'encoding':X[0],
		'prediction':pred,
	})

	print('prediction:')
	for i, op in enumerate(OPERATOR_LIST):
		print('\t', op, ':', pred[i])

	if projected:
		att = np.zeros((len(OPERATOR_LIST), X.shape[1]))
		for i in range(len(OPERATOR_LIST)):
			att[i] = get_att_for_op(
				decoder, X, i, nvar=nvar,
				path_template=args.decoder_path_template
			)
		write_h5_file(args.path, {'attention':att})

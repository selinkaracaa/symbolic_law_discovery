from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from decoder.run_decoder import predict_for_op
from decoder.train_decoder import train_decoder
from encoder.MLP.run_encoder import encode as MLP_encode
from encoder.super_resolution.prepare_dataset import get_naive_encodings, get_encodings
from encoder.super_resolution.run_encoder import encode as SR_encoder
from encoder.super_resolution.train_encoder import train_super_resolution
from model.CNN import CNN
from model.MIL_Attention import MIL_Attention
from model.super_resolution import EDSR


def train_model(
	inp_train, Y_train, funcs_train, inp_val, Y_val, funcs_val,
	batch_size=BATCH_SIZE, encoding_scheme=ENCODING_SCHEME, 
	encoder_path=SUPERR_ENCODER_WEIGHT_PATH, 
	decoder_path_template=DECODER_WEIGHT_PATH_TEMPLATE):

	projected = len(inp_train.shape) == 4
	nvar = inp_train.shape[-1] - 1
	ndim = 2

	if encoding_scheme == 'MLP':
		X_train = MLP_encode(inp_train[:])
		X_val = MLP_encode(inp_val[:])
	elif encoding_scheme == 'super resolution':
		lr_train, hr_train = get_encodings(inp_train, funcs_train, nvar=nvar, ndim=ndim)
		lr_val, hr_val = get_encodings(inp_val, funcs_val, nvar=nvar, ndim=ndim)

	encoder = train_super_resolution(
		lr_train, hr_train, lr_val, hr_val, batch_size=batch_size)
	encoder.save_weights(encoder_path)

	X_train = SR_encoder(encoder, lr_train, batch_size=batch_size)
	X_val = SR_encoder(encoder, lr_val, batch_size=batch_size)

	for i in range(len(OPERATOR_LIST)):
		decoder = train_decoder(X_train, Y_train[:, i], X_val, Y_val[:, i])
		decoder.save_weights(decoder_path_template.format(nvar, i))


def get_prediction(
	inp, batch_size=BATCH_SIZE, encoding_scheme=ENCODING_SCHEME, 
	encoder_path=SUPERR_ENCODER_WEIGHT_PATH, 
	decoder_path_template=DECODER_WEIGHT_PATH_TEMPLATE):

	projected = len(inp.shape) == 4
	nvar = inp.shape[-1] - 1
	ndim = 2

	if encoding_scheme == 'MLP':
		X = MLP_encode(inp[:])
	elif encoding_scheme == 'super resolution':
		encoder = EDSR()
		encoder.load_weights(encoder_path)
		lr = get_naive_encodings(inp)
		X = SR_encoder(encoder, lr, batch_size=batch_size)

	if projected:
		decoder = MIL_Attention()
	else:
		decoder = CNN()

	pred = np.zeros((len(inp), len(OPERATOR_LIST)))

	for i in range(len(OPERATOR_LIST)):
		pred[:, i] = predict_for_op(decoder, X, i, nvar=nvar)

	return pred
import argparse
import h5py

from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from pipeline.pipelines import train_model


parser = argparse.ArgumentParser(description='Train Encoder and Decoder Models')

parser.add_argument('--path', default=DATA_PATH, 
	help='path of training data')
parser.add_argument('--encoder', default=ENCODING_SCHEME, 
	help='encoding scheme used')
parser.add_argument('--encoder_path', default=SUPERR_ENCODER_WEIGHT_PATH, 
	help='path for the super-resolution encoder')
parser.add_argument('--decoder_path_template', default=DECODER_WEIGHT_PATH_TEMPLATE, 
	help='path template for the decoder')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)


args = parser.parse_args()

f = h5py.File(args.path, 'r')

Y_train = f['Y_train'][:]
Y_val = f['Y_val'][:]
funcs_train = f['funcs_train']
funcs_val = f['funcs_val']

nvar = f['inp_train'].shape[-1] - 1
ndim = 2
projected = 'inp_proj_train' in f.keys()

if projected:
	inp_train = f['inp_proj_train']
	inp_val = f['inp_proj_val']
else:
	inp_train = f['inp_train']
	inp_val = f['inp_val']

train_model(
	inp_train, Y_train, funcs_train, inp_val, Y_val, funcs_val,
	batch_size=args.batch_size, encoding_scheme=args.encoder, 
	encoder_path=args.encoder_path, 
	decoder_path_template=args.decoder_path_template,
)

print()
print("Encoder weights saved to:", args.encoder_path)
print("Decoder weights saved to:")
for i, op in enumerate(OPERATOR_LIST):
	print('\t', op, ':', args.decoder_path_template.format(nvar, i))
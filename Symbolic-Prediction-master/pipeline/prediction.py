import argparse
import h5py
from scipy.integrate import trapezoid as trapz
from sklearn.metrics import accuracy_score, roc_curve
from tabulate import tabulate

from common.preset import *
from common.symbolic_tree import OPERATOR_LIST
from common.write_h5_file import write_h5_file
from pipeline.pipelines import get_prediction


parser = argparse.ArgumentParser(description='Predict with Encoder and Decoder Models')

parser.add_argument('--path', default=DATA_PATH, 
	help='data path')
parser.add_argument('--encoder', default=ENCODING_SCHEME, 
	help='encoding scheme used')
parser.add_argument('--encoder_path', default=SUPERR_ENCODER_WEIGHT_PATH, 
	help='path for the super-resolution encoder')
parser.add_argument('--decoder_path_template', default=DECODER_WEIGHT_PATH_TEMPLATE, 
	help='path template for the decoder')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)


args = parser.parse_args()


def print_performance(acc, auc):
	tab_data = [[op, *acc[:, i], auc[i]] for i, op in enumerate(OPERATOR_LIST)]
	print(tabulate(tab_data, headers=['Op', 'Train Acc', 'Val Acc', 'Test Acc', 'Test Auc']))

f = h5py.File(args.path, 'a')

Y_train = f['Y_train'][:]
Y_val = f['Y_val'][:]
Y_test = f['Y_test'][:]


nvar = f['inp_train'].shape[-1] - 1
ndim = 2
projected = 'inp_proj_train' in f.keys()

if projected:
	inp_train = f['inp_proj_train']
	inp_val = f['inp_proj_val']
	inp_test = f['inp_proj_test']
else:
	inp_train = f['inp_train']
	inp_val = f['inp_val']
	inp_test = f['inp_test']

pred_train = get_prediction(
	inp_train,
	batch_size=args.batch_size, encoding_scheme=args.encoder, 
	encoder_path=args.encoder_path, 
	decoder_path_template=args.decoder_path_template
)
pred_val = get_prediction(
	inp_val,
	batch_size=args.batch_size, encoding_scheme=args.encoder, 
	encoder_path=args.encoder_path, 
	decoder_path_template=args.decoder_path_template
)
pred_test = get_prediction(
	inp_test,
	batch_size=args.batch_size, encoding_scheme=args.encoder, 
	encoder_path=args.encoder_path, 
	decoder_path_template=args.decoder_path_template
)

acc = np.zeros((3, len(OPERATOR_LIST)))
auc = np.zeros(len(OPERATOR_LIST))

for i in range(len(OPERATOR_LIST)):
	acc[0, i] = accuracy_score(pred_train[:, i] > 0.5, Y_train[:, i])
	acc[1, i] = accuracy_score(pred_val[:, i] > 0.5, Y_val[:, i])
	acc[2, i] = accuracy_score(pred_test[:, i] > 0.5, Y_test[:, i])
	fpr, tpr, thresholds = roc_curve(Y_test[:, i], pred_test[:, i])
	auc[i] = trapz(tpr, x=fpr)

f.close()
print()
print_performance(acc, auc)
print()
print('Predictions saved to', args.path)
write_h5_file(args.path, {
	'prediction_train': pred_train,
	'prediction_val': pred_val,
	'prediction_test': pred_test,
})
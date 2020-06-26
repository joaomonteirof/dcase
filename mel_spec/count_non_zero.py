from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets
from models import vgg, resnet, densenet, base_cnn, TDNN
import numpy as np
from utils import countNonZeroWeights, get_data

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Count non zero params')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data just to get num of classes')
	parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet', 'tdnn'], default='resnet')
	args = parser.parse_args()

	testset = datasets.DatasetFolder(root=args.data_path, loader=get_data, extensions=('wav'))
	args.nclasses = len(testset.classes)

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	if args.model == 'cnn':
		model = base_cnn.CNN(n_classes=args.nclasses)
	elif args.model == 'vgg':
		model = vgg.VGG('VGG11', n_classes=args.nclasses)
	elif args.model == 'resnet':
		model = resnet.ResNet12(n_classes=args.nclasses)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(n_classes=args.nclasses)
	elif args.model == 'tdnn':
		model = TDNN.TDNN(n_classes=args.nclasses)
	
	try:
		print(model.load_state_dict(ckpt['model_state'], strict=True))
		print('\n')
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

	print('\n\nNumber of parameters: {}\n'.format(sum(p.numel() for p in model.parameters())))

	total_params, nonzero_params = sum(p.numel() for p in model.parameters()), countNonZeroWeights(model)

	assert total_params > nonzero_params, 'Error while computing number of non zero params!!! - Total-Nonzero={}'.format(total_params - nonzero_params)

	model_size = nonzero_params*32./1000 + (total_params-nonzero_params)*2./1000 ## kb

	print('Total parameters: {}'.format(total_params))
	print('Nonzero parameters: {}'.format(nonzero_params))
	print('Zero parameters: {}'.format((total_params-nonzero_params)))
	print('Model size: {}kb'.format(model_size))
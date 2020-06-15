from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import vgg, resnet, densenet, base_cnn, TDNN
from sklearn.metrics import confusion_matrix, log_loss
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import get_data_evaluation, parse_args_for_log, get_freer_gpu

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-path', type=str, default='./out.csv', metavar='Path', help='Path to output scores')
	parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet', 'tdnn'], default='resnet')
	parser.add_argument('--aux-data', type=str, default='./aux_data/', metavar='Path', help='Aux data to get classes list')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=0, metavar='N', help='Data load workers (default: 0)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	auxset = datasets.DatasetFolder(root=args.aux_data, loader=get_data_evaluation, extensions=('wav'))
	testset = datasets.DatasetFolder(root=args.data_path, loader=get_data_evaluation, extensions=('wav'))

	args.nclasses = len(auxset.classes)

	args_dict = parse_args_for_log(args)
	print('\n')
	for key in args_dict:
		print('{}: {}'.format(key, args_dict[key]))
	print('\n')

	idx_to_class = {}

	for key in auxset.class_to_idx:
		idx_to_class[str(auxset.class_to_idx[key])] = key
	print(idx_to_class, '\n')

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

	if args.cuda:
		device = get_freer_gpu()
		model = model.to(device)
	else:
		device=torch.device('cpu')

	model.eval()

	out_data = [['filename', 'scene_label', *auxset.classes]]

	with torch.no_grad():

		iterator = tqdm(testset, total=len(test_loader))
		for batch in iterator:

			filename, data, _ = batch

			filename.split('/')[-1]

			x = data.to(device).unsqueeze(0)

			out = F.softmax(model.forward(x), dim=1)

			scores = {}

			for index in idx_to_class:
				scores[idx_to_class[index]] = out[0, int(index)].item()

			pred_idx = str(out.max(1)[1].long().tem())
			pred = idx_to_class[pred_idx]

			out_data.append([filename, pred, *[score[class_name] for class_name in auxset.classes]])


	print('Storing scores in output file:')
	print(args.out_path)

	with open(args.out_path, 'w') as f:
		for line in out_data:
			f.write("%s" % '\t'.join(*line)+'\n')
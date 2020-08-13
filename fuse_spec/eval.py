from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import resnet
from sklearn.metrics import confusion_matrix, log_loss
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import get_data, parse_args_for_log, get_freer_gpu, set_np_randomseed

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['resnet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	testset = datasets.DatasetFolder(root=args.data_path, loader=get_data, extensions=('wav'))
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	args.nclasses = len(testset.classes)

	args_dict = parse_args_for_log(args)
	print('\n')
	for key in args_dict:
		print('{}: {}'.format(key, args_dict[key]))
	print('\n')

	idx_to_class = {}

	for key in testset.class_to_idx:
		idx_to_class[str(testset.class_to_idx[key])] = key
	print(idx_to_class, '\n')

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	if args.model == 'resnet':
		model = resnet.ResNet12(n_classes=args.nclasses)
	
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

	predictions = []
	labels = []
	scores = []

	with torch.no_grad():

		iterator = tqdm(test_loader, total=len(test_loader))
		for batch in iterator:

			x_1, x_2, y = batch

			x_1 = x_1.to(device)
			x_1 = x_1.to(device)
			y = y.to(device)

			out = F.softmax(model.forward(x_1, x_2), dim=1)

			pred = out.max(1)[1].long()

			predictions.append(pred)
			labels.append(y)
			scores.append(out)

		predictions = torch.cat(predictions, 0).cpu().numpy()
		labels = torch.cat(labels, 0).cpu().numpy()
		scores = torch.cat(scores, 0).cpu()

	classes_list = testset.classes
	cm_matrix = confusion_matrix(labels, predictions)
	accuracies = 100.0*cm_matrix.diagonal()/cm_matrix.sum(axis=1)

	print('\nAccuracies - Log loss:\n')
	for i, class_ in enumerate(classes_list):
		print(class_, ': {:0.4f}% - {:0.4f}'.format(accuracies[i], F.binary_cross_entropy(input=scores[:, testset.class_to_idx[class_]], target=torch.where(torch.from_numpy(labels) == testset.class_to_idx[class_], torch.ones(scores.size(0)), torch.zeros(scores.size(0)))).item()))

	print('\n')
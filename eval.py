from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import vgg, resnet, densenet, base_cnn
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import MEAN, STD, get_data, parse_args_for_log, get_freer_gpu, set_np_randomseed

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform = transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
	testset = datasets.DatasetFolder(root=args.data_path, loader=get_data, transform=transform, extensions=('mat'))
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

	if args.model == 'cnn':
		model = base_cnn.CNN(n_classes=args.nclasses)
	elif args.model == 'vgg':
		model = vgg.VGG('VGG16', n_classes=args.nclasses)
	elif args.model == 'resnet':
		model = resnet.ResNet18(n_classes=args.nclasses)
	elif args.model == 'densenet':
		model = densenet.DenseNet121(n_classes=args.nclasses)
	
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

	model.eval()

	predictions = []
	labels = []

	with torch.no_grad():

		iterator = tqdm(test_loader, total=len(test_loader))
		for batch in iterator:

			x, y = batch

			x = x.to(device)

			out = model.forward(x)

			pred = F.softmax(out, dim=1).max(1)[1].long()

			predictions.append(pred)
			labels.append(y)

		predictions = torch.cat(predictions, 0).cpu().numpy()
		labels = torch.cat(labels, 0).cpu().numpy()


	cm_matrix = confusion_matrix(labels, predictions)
	accuracies = 100.0*cm_matrix.diagonal()/cm_matrix.sum(axis=1)

	for i, acc in enumerate(accuracies):
		print('\n')
		print(idx_to_class[str(i)], ': {:0.4f}%'.format(acc))

	print('\n')
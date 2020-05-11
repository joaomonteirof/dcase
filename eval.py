from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import vgg, resnet, densenet
 from sklearn.metrics import confusion_matrix
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import MEAN, STD, get_data

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
	parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	transform = transforms.Compose([transforms.Normalize(mean=MEAN, std=STD)])
	dataset = datasets.DatasetFolder(root=args.data_path, loader=get_data, transform=transform, extensions=('mat'))
	test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	for key in testset.class_to_idx:
		idx_to_class[str(testset.class_to_idx)] = key

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	if args.model == 'vgg':
		model = vgg.VGG('VGG16', nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'resnet':
		model = resnet.ResNet18(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	elif args.model == 'densenet':
		model = densenet.densenet_cifar(nh=n_hidden, n_h=hidden_size, dropout_prob=dropout_prob, sm_type=softmax)
	
	try:
		model.load_state_dict(ckpt['model_state'], strict=True)
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

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
			labels.append(labels)

		predictions = torch.cat(predictions, 0).numpy()
		labels = torch.cat(labels, 0).numpy()


	cm_matrix = confusion_matrix(labels, predictions)
	accuracies = cm_matrix.diagonal()/cm_matrix.sum(axis=1)

	for i, acc in enumerate(accuracies):
		print('\n')
		print(idx_to_class[str(i)], acc)
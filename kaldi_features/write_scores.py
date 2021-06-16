from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import h5py
from torchvision import datasets, transforms
from models import vgg, resnet, densenet, base_cnn, TDNN
from sklearn.metrics import confusion_matrix, log_loss
from data_load import Loader
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import *


def prep_utterance(data, max_nb_frames=):

		if data.shape[-1]>max_nb_frames:
			ridx = np.random.randint(0, data.shape[-1]-max_nb_frames)
			data_ = data[:, :, ridx:(ridx+max_nb_frames)]
		else:
			mul = int(np.ceil(max_nb_frames/data.shape[-1]))
			data_ = np.tile(data, (1, 1, mul))
			data_ = data_[:, :, :max_nb_frames]

		return torch.from_numpy(data_).contiguous()


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Evaluation')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
	parser.add_argument('--data-path', type=str, default='./data/data.hdf', metavar='Path', help='Path to data')
	parser.add_argument('--n-frames', type=int, default=500, metavar='N', help='Number of frames per utterance (default: 500)')
	parser.add_argument('--batch-size', type=int, default=64, metavar='B', help='Batch size (default: 64)')
	parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet', 'tdnn'], default='resnet')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--workers', type=int, default=4, metavar='N', help='Data load workers (default: 4)')
	parser.add_argument('--out-path', type=str, default='./scores.out', metavar='Path', help='Path to output scores')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	testset = h5py.File(args.data_path, 'r')

	args_dict = parse_args_for_log(args)
	print('\n')
	for key in args_dict:
		print('{}: {}'.format(key, args_dict[key]))
	print('\n')

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

	outputs = []

	with torch.no_grad():

		for spk in testset:
			for spk_utt in testset[spk]:

				x = prep_utterance(testset[spk][spk_utt])
				x = x.to(device)

				out = str(F.softmax(model.forward(x), dim=1).cpu().numpy())

				outputs.append(f"{spk_utt} {out}\n")


	print('\n')

	with open(args.out_path, 'w') as f:
		for spk_scores in outputs:
			f.write(spk_scores)
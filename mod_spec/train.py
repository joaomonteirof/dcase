from __future__ import print_function
import argparse
import torch
import torchvision
from torch.utils.data import DataLoader
from train_loop import TrainLoop
import torch.optim as optim
from torchvision import datasets, transforms
from models import vgg, resnet, densenet, base_cnn, TDNN
import numpy as np
from time import sleep
import os
import sys
from utils import MEAN, STD, get_data, parse_args_for_log, get_freer_gpu, set_np_randomseed, augment

# Training settings
parser = argparse.ArgumentParser(description='Acoustic scene classification from modulation spectra')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=16, metavar='N', help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='m', help='Momentum paprameter (default: 0.9)')
parser.add_argument('--patience', type=int, default=30, metavar='N', help='number of epochs to wait whith no improvement prior to reducing lr')
parser.add_argument('--l2', type=float, default=1e-4, metavar='lambda', help='L2 wheight decay coefficient (default: 0.0005)')
parser.add_argument('--smoothing', type=float, default=0.2, metavar='l', help='Label smoothing (default: 0.2)')
parser.add_argument('--max-gnorm', type=float, default=10., metavar='clip', help='Max gradient norm (default: 10.0)')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--data-path', type=str, default='./data_train', metavar='Path', help='Path to data')
parser.add_argument('--valid-data-path', type=str, default='./data_val', metavar='Path', help='Path to data')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--n-workers', type=int, default=4, metavar='N', help='Workers for data loading. Default is 4')
parser.add_argument('--model', choices=['cnn', 'vgg', 'resnet', 'densenet', 'tdnn'], default='resnet')
parser.add_argument('--pretrained-path', type=str, default=None, metavar='Path', help='Path to trained model. Discards outpu layer')
parser.add_argument('--save-every', type=int, default=1, metavar='N', help='how many epochs to wait before saving checkpoints. Default is 1')
parser.add_argument('--eval-every', type=int, default=1000, metavar='N', help='how many iterations to wait before evaluatiing models. Default is 1000')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--no-aug', action='store_true', default=False, help='Disables data augmentation')
parser.add_argument('--no-cp', action='store_true', default=False, help='Disables checkpointing')
parser.add_argument('--verbose', type=int, default=1, metavar='N', help='Verbose is activated if > 0')
parser.add_argument('--logdir', type=str, default=None, metavar='Path', help='Path for checkpointing')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

if args.cuda:
	torch.backends.cudnn.benchmark=True

transform = None if  args.no_aug else augment

trainset = datasets.DatasetFolder(root=args.data_path, loader=get_data, transform=transform, extensions=('mat'))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, worker_init_fn=set_np_randomseed, pin_memory=True)

validset = datasets.DatasetFolder(root=args.valid_data_path, loader=get_data, extensions=('mat'))
valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

args.nclasses = len(trainset.classes)

print(args, '\n')

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

if args.pretrained_path:
	try:
		print('\nLoading pretrained model from: {}\n'.format(args.pretrained_path))
		ckpt=torch.load(args.pretrained_path, map_location = lambda storage, loc: storage)
		print(model.load_state_dict(ckpt['model_state'], strict=False))
		print('\n')
	except RuntimeError as err:
		print("Runtime Error: {0}".format(err))
	except:
		print("Unexpected error:", sys.exc_info()[0])
		raise

if args.verbose >0:
	print(model)
	print('\n\nNumber of parameters: {}\n'.format(sum(p.numel() for p in model.parameters())))

if args.cuda:
	device = get_freer_gpu()
	model = model.to(device)

if args.logdir:
	from torch.utils.tensorboard import SummaryWriter
	writer = SummaryWriter(log_dir=args.logdir, comment=args.model, purge_step=True if args.checkpoint_epoch is None else False)
	args_dict = parse_args_for_log(args)
	writer.add_hparams(hparam_dict=args_dict, metric_dict={'best_acc':0.0})
else:
	writer = None
	args_dict = None

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)

trainer = TrainLoop(model, optimizer, train_loader, valid_loader, max_gnorm=args.max_gnorm, label_smoothing=args.smoothing, verbose=args.verbose, save_cp=(not args.no_cp), checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, patience=args.patience, cuda=args.cuda, logger=writer)

if args.verbose >0:
	if args_dict is None:
		args_dict = parse_args_for_log(args)
	print('\n')
	for key in args_dict:
		print('{}: {}'.format(key, args_dict[key]))
	print('\n')

trainer.train(n_epochs=args.epochs, save_every=args.save_every, eval_every=args.eval_every)
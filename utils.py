import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn import metrics
import scipy.io as sio
import torch
import itertools
import os
import sys
import pickle
from time import sleep
import random

MEAN = -34.5485
STD = 17.5655

def parse_args_for_log(args):
	args_dict = dict(vars(args))
	for arg_key in args_dict:
		if args_dict[arg_key] is None:
			args_dict[arg_key] = 'None'

	return args_dict

def set_np_randomseed(worker_id):
	np.random.seed(np.random.get_state()[1][0]+worker_id)

def get_freer_gpu(trials=10):
	sleep(2)
	for j in range(trials):
		os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
		try:
			a = torch.rand(1).cuda(dev_)
			return dev_
		except:
			pass

	print('NO GPU AVAILABLE!!!')
	exit(1)

def augment(example):

	with torch.no_grad():

		if random.rand()>0.5:
			example = freq_mask(example, dim=1)
		if random.rand()>0.5:
			example = freq_mask(example, dim=2)
		if random.rand()>0.5:
			example += torch.randn_like(example)*random.choice([1e-2, 1e-3, 1e-4, 1e-5])

	return example

def get_data(path):
	data = sio.loadmat(path)
	data = data[sorted(data.keys())[0]]
	data = torch.Tensor(data).float().unsqueeze(0).contiguous()

	return data

def adjust_learning_rate(optimizer, epoch, base_lr, n_epochs, lr_factor, min_lr=1e-8):
	"""Sets the learning rate to the initial LR decayed by 10 every n_epochs epochs"""
	lr = max( base_lr * (lr_factor ** (epoch // n_epochs)), min_lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def correct_topk(output, target, topk=(1,)):
	"""Computes the number of correct predicitions over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k)
		return res

def freq_mask(spec, F=30, num_masks=1, replace_with_zero=False, dim=1):
	"""Frequency masking

	adapted from https://espnet.github.io/espnet/_modules/espnet/utils/spec_augment.html

	:param torch.Tensor spec: input tensor with shape (T, dim)
	:param int F: maximum width of each mask
	:param int num_masks: number of masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0,
		if False, filled with mean
	:param int dim: 1 or 2 indicating to which axis the mask corresponds
	"""

	with torch.no_grad():

		cloned = spec.clone()

		for i in range(0, num_masks):
			f = random.randrange(0, F)
			f_zero = random.randrange(0, 1 - f)

			# avoids randrange error if values are equal and range is empty
			if f_zero == f_zero + f:
				return cloned

			mask_end = random.randrange(f_zero, f_zero + f)
			if replace_with_zero:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = 0.0
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = 0.0
			else:
				if dim==1:
					cloned[:, f_zero:mask_end, :] = cloned.mean()
				elif dim==2:
					cloned[:, :, f_zero:mask_end] = cloned.mean()

	return cloned
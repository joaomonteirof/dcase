import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn import metrics
import scipy.io as sio
import torch
import torchaudio
import itertools
import os
import sys
import pickle
from time import sleep
import random
from scipy.signal import convolve2d
import argparse
from tempfile import NamedTemporaryFile

def collater(batch):

	spec, mod, labels = [], [], []

	for item in batch:
		spec.append(item[0][0].unsqueeze(0))
		mod.append(item[0][1].unsqueeze(0))
		labels.append(torch.Tensor([item[1]]))

	spec, mod, labels = torch.cat(spec, dim=0).float().contiguous(), torch.cat(mod, dim=0).float().contiguous(), torch.cat(labels, dim=0).long()

	return spec, mod, labels

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

def load_audio(path):
	sound, _ = torchaudio.load(path, normalization=True)
	sound = sound.numpy().T
	if len(sound.shape) > 1:
		if sound.shape[1] == 1:
			sound = sound.squeeze()
		else:
			sound = sound.mean(axis=1)  # multiple channels, average

	return sound

def augment_audio(path, sample_rate, tempo, gain):
	"""
	Changes tempo and gain of the recording with sox and loads it.
	sudo apt-get update && sudo apt-get install sox libsox-fmt-all
	"""
	with NamedTemporaryFile(suffix=".wav") as augmented_file:
		augmented_filename = augmented_file.name
		sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
		sox_params = "sox -t wav \"{}\" -r {} -c 1 -b 24 -e signed {} {} >/dev/null 2>&1".format(path, sample_rate, augmented_filename, " ".join(sox_augment_params))
		os.system(sox_params)
		if random.random()>0.8:
			sox_params = "sox -t wav \"{}\" -r {} -c 1 -b 24 -e signed reverse >/dev/null 2>&1".format(path, sample_rate, augmented_filename)
			os.system(sox_params)
		y = load_audio(augmented_filename)
		return y

def load_randomly_augmented_audio(path, sample_rate=48000, tempo_range=(0.7, 1.3), gain_range=(-7, 9)):
	"""
	Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
	Returns the augmented utterance.
	"""
	low_tempo, high_tempo = tempo_range
	tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
	low_gain, high_gain = gain_range
	gain_value = np.random.uniform(low=low_gain, high=high_gain)
	audio = augment_audio(path=path, sample_rate=sample_rate, tempo=tempo_value, gain=gain_value)
	return audio

def augment_spec(example):

	with torch.no_grad():

		if random.random()>0.5:
			example = freq_mask(example, F=40, dim=1)
		if random.random()>0.5:
			example = freq_mask(example, F=32, dim=2)
		if random.random()>0.5:
			example += torch.randn_like(example)*random.choice([1e-1, 1e-2, 1e-3])

	return example

def normalize(example):

	with torch.no_grad():

		mean = example.mean(dim=-1, keepdim=True) #convolve2d(example, np.ones([1, 100]), mode='same')/100.
		example -= mean

	return example

def compute_features(audio):

	audio = torch.from_numpy(audio).unsqueeze(0)

	spec = torchaudio.compliance.kaldi.fbank(audio, frame_length=40, frame_shift=20, num_mel_bins=40, sample_frequency=48000, high_freq=22050, low_freq=0, use_log_fbank=True).T

	if spec.shape[-1]>=500:
		spec = spec[:,:500]
	else:
		spec = spec.repeat(1,500//spec.shape[-1]+1)[:,:500]

	modspec = torch.stft(input=spec, n_fft=32, hop_length=16, win_length=32, center=True, pad_mode='reflect', normalized=False, onesided=False)

	modspec = torch.sqrt(torch.pow(modspec[...,0], 2)+torch.pow(modspec[...,1], 2)) ## get absolute value

	modspec = modspec.permute(2, 0, 1) ## averaging out the time dim

	spec = spec.unsqueeze(0).float().contiguous()
	modspec = modspec.float().contiguous()

	return spec, modspec

def get_data(path):

	data = load_audio(path)

	data_spec, data_mod = compute_features(data)

	return data_spec, data_mod

def get_data_augment(path):

	if random.random()>0.5:
		data = load_randomly_augmented_audio(path)
	else:
		data = load_audio(path)

	data_spec, data_mod = compute_features(data)

	if random.random()>0.5:
		data_spec, data_mod = augment_spec(data_spec), augment_spec(data_mod)

	return data_spec, data_mod

def get_data_evaluation(path):

	data = load_audio(path)

	data_spec, data_mod = compute_features(data)

	return path, data_spec, data_mod

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

def freq_mask(spec, F=100, num_masks=1, replace_with_zero=False, dim=1):
	"""Frequency masking

	adapted from https://espnet.github.io/espnet/_modules/espnet/utils/spec_augment.html

	:param torch.Tensor spec: input tensor with shape (T, dim)
	:param int F: maximum width of each mask
	:param int num_masks: number of masks
	:param bool replace_with_zero: if True, masked parts will be filled with 0,
		if False, filled with mean
	:param int dim: 1 or 2 indicating to which axis the mask corresponds
	"""

	assert dim==1 or dim==2, 'Only 1 or 2 are valid values for dim!'

	with torch.no_grad():

		cloned = spec.clone()
		num_bins = cloned.shape[dim]

		for i in range(0, num_masks):
			f = random.randrange(0, F)
			f_zero = random.randrange(0, num_bins - f)

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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Test data load and augment')
	parser.add_argument('--file-path', type=str, default=None, metavar='Path', help='wav sample for testing')
	args = parser.parse_args()

	testing_data_spec, testing_data_mod = get_data(args.file_path)
	print(type(testing_data_spec), type(testing_data_spec))
	print(testing_data_spec.shape, testing_data_spec.max(), testing_data_spec.min())
	print(testing_data_mod.shape, testing_data_mod.max(), testing_data_mod.min())

	testing_data_spec, testing_data_mod = get_data_augment(args.file_path)
	print(type(testing_data_spec), type(testing_data_spec))
	print(testing_data_spec.shape, testing_data_spec.max(), testing_data_spec.min())
	print(testing_data_mod.shape, testing_data_mod.max(), testing_data_mod.min())
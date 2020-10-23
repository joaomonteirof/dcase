import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class Loader(Dataset):

	def __init__(self, hdf5_name):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt, spk, y = self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = torch.from_numpy( self.open_file[spk][utt][()] )

		return utt_data.contiguous(), y.squeeze()

	def __len__(self):
		return len(self.utt_list)

	def create_lists(self):

		open_file = h5py.File(self.hdf5_name, 'r')

		self.spk_list = sorted(list(open_file.keys()))
		self.spk2label = {}
		self.spk2utt = {}

		for i, spk in enumerate(self.spk_list):
			spk_utt_list = list(open_file[spk])
			self.spk2utt[spk] = spk_utt_list
			self.spk2label[spk] = torch.LongTensor([i])

		open_file.close()

		self.nclasses = len(self.spk2utt)

	def update_lists(self):

		self.utt_list = []

		for i, spk in enumerate(self.spk2utt):
			spk_utt_list = np.random.permutation(list(self.spk2utt[spk]))
			for utt in spk_utt_list:
				self.utt_list.append([utt, spk, self.spk2label[spk]])

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from librosa.feature import delta as delta_

class Loader(Dataset):

	def __init__(self, hdf5_name, max_nb_frames=2000, train=True, delta=False):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name
		self.max_nb_frames = int(max_nb_frames)
		self.delta=delta
		self.train=train

		self.create_lists()

		self.open_file = None

		self.update_lists()

	def __getitem__(self, index):

		utt, spk, y = self.utt_list[index]

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		utt_data = torch.from_numpy( self.prep_utterance( self.open_file[spk][utt] ) )

		return utt_data.contiguous(), y.squeeze()

	def __len__(self):
		return len(self.utt_list)

	def prep_utterance(self, data):

		if self.train:

			if data.shape[-1]>self.max_nb_frames:
				ridx = np.random.randint(0, data.shape[-1]-self.max_nb_frames)
				data_ = data[:, :, ridx:(ridx+self.max_nb_frames)]
			else:
				mul = int(np.ceil(self.max_nb_frames/data.shape[-1]))
				data_ = np.tile(data, (1, 1, mul))
				data_ = data_[:, :, :self.max_nb_frames]

			if self.delta:
				data_ = np.concatenate([data_, delta_(data_,width=3,order=1), delta_(data_,width=3,order=2)], axis=0)

		return data_

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

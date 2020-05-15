import torch
import torch.nn as nn
import torch.nn.functional as F

class StatisticalPooling(nn.Module):

	def forward(self, x):
		# x is 3-D with axis [B, feats, T]
		mu = x.mean(dim=2, keepdim=False)
		std = (x+torch.randn_like(x)*1e-6).std(dim=2, keepdim=False)
		return torch.cat((mu, std), dim=1)

class TDNN(nn.Module):
	def __init__(self, ncoef=257, init_coef=0, n_classes=1000):
		super(TDNN, self).__init__()

		self.ncoef=ncoef
		self.init_coef=init_coef
		self.n_classes = n_classes

		self.model = nn.Sequential( nn.Conv1d(ncoef, 512, 5, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=2, padding=2),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 3, dilation=3, padding=3),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 1500, 1),
			nn.BatchNorm1d(1500),
			nn.ReLU(inplace=True) )

		self.pooling = StatisticalPooling()

		self.post_pooling = nn.Sequential(nn.Linear(3000, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Linear(512, self.n_classes) )

	def forward(self, x):

		x = x[:,:,self.init_coef:,:].squeeze(1)

		x = self.model(x)
		x = self.pooling(x)
		out = self.post_pooling(x)

		return out
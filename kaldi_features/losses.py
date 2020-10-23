import math
import torch
from torch import nn
from scipy.special import binom
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
	def __init__(self, label_smoothing, lbl_set_size, dim=1):
		super(LabelSmoothingLoss, self).__init__()
		self.confidence = 1.0 - label_smoothing
		self.smoothing = label_smoothing
		self.cls = lbl_set_size
		self.dim = dim

	def forward(self, pred, target):
		pred = pred.log_softmax(dim=self.dim)
		with torch.no_grad():
			# true_dist = pred.data.clone()
			true_dist = torch.zeros_like(pred)
			true_dist.fill_(self.smoothing / (self.cls - 1))
			true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

		return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

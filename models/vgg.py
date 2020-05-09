import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.losses import AMSoftmax, Softmax

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, n_classes=1000):
		super(VGG, self).__init__()

		self.n_classes = n_classes

		self.features = self._make_layers(cfg[vgg_name])
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.lin_proj = nn.Sequential(nn.Linear(512 * 7 * 7, n_classes))

	def forward(self, x):
		x = self.avgpool(self.features(x))
		x = x.view(x.size(0), -1)
		x = self.lin_proj(x)

		return x

	def _make_layers(self, cfg):
		layers = []
		in_channels = 1
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)
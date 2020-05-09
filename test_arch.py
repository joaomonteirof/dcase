from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import vgg, resnet, densenet

# Training settings
parser = argparse.ArgumentParser(description='Test architectures with dummy data')
parser.add_argument('--model', choices=['vgg', 'resnet', 'densenet'], default='resnet')
parser.add_argument('--nclasses', type=int, default=10, metavar='N', help='number of classes')
args = parser.parse_args()

if args.model == 'vgg':
	model = vgg.VGG('VGG19', n_classes=args.nclasses)
elif args.model == 'resnet':
	model = resnet.ResNet50(n_classes=args.nclasses)
elif args.model == 'densenet':
	model = densenet.DenseNet121(n_classes=args.nclasses)

batch = torch.rand(3, 1, 257, 257)

out = model.forward(batch)

print(out.size())
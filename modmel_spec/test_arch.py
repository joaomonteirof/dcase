from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
from models import resnet

# Training settings
parser = argparse.ArgumentParser(description='Test architectures with dummy data')
parser.add_argument('--model', choices=['resnet'], default='resnet')
parser.add_argument('--half-spec', action='store_true', default=False, help='Discards the lower frequency half of the spectrum')
parser.add_argument('--nclasses', type=int, default=10, metavar='N', help='number of classes')
args = parser.parse_args()

if args.model == 'resnet':
	model = resnet.ResNet18(n_classes=args.nclasses, half_spec=args.half_spec)

print('\n', model, '\n')
print('\n\nNumber of parameters: {}\n'.format(sum(p.numel() for p in model.parameters())))

batch = torch.rand(3, 32, 40, 32)

out = model.forward(batch)

print(out.size(), '\n')
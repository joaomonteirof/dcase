from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import parse_csv
import glob

def get_header(path):
	with open(path, 'r') as file:
		data = file.readlines()

	return data[0].strip().split('\t')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Score level fusion')
	parser.add_argument('--data-path', type=str, default='./data/', metavar='Path', help='Path to data')
	parser.add_argument('--out-path', type=str, default='./out.csv', metavar='Path', help='Path to output scores')
	args = parser.parse_args()

	file_list = glob.glob(args.data_path + '*.csv')

	print(file_list)

	assert len(file_list)>1, 'Not enough files found in the specified folder. At least two files with score should be available in the folder.'

	score_files = []

	for score_file in file_list:
		score_files.append(parse_csv(score_file))

	out_data = [get_header(file_list[0])]

	classes = out_data[0][2:]

	idx_to_class = {}
	for i, clss in enumerate(classes):
		idx_to_class[str(i)] = clss

	with torch.no_grad():

		iterator = tqdm(score_files[0], total=len(score_files))
		for filename in iterator:

			out = 0.0

			for score_dict in score_files:
				out += score_dict[filename]

			out = F.softmax(out, dim=1)

			scores = {}

			for index in idx_to_class:
				scores[idx_to_class[index]] = out[0, int(index)].item()

			pred_idx = str(out.max(1)[1].long().item())
			pred = idx_to_class[pred_idx]

			out_data.append([filename, pred, *[str(scores[class_name]) for class_name in classes]])


	print('Storing scores in output file:')
	print(args.out_path)

	with open(args.out_path, 'w') as f:
		for line in out_data:
			f.write("%s" % '\t'.join(line)+'\n')
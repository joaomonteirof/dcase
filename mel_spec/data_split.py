import argparse
import os
import pathlib
from shutil import copyfile

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Get data stats')
	parser.add_argument('--wav-data', type=str, default='./wav/', metavar='Path', help='Path to wav files organized in one folder per class')
	parser.add_argument('--split-data', type=str, default='./mat/', metavar='Path', help='Path to mat files split in advance')
	args = parser.parse_args()

	wav_list = glob.glob(args.wav_data+'/**/*.wav', recursive = True)

	train_mat_files = pathlib.Path(os.path.join(args.wav_data, '/train')).mkdir(parents=True, exist_ok=True)
	test_mat_files = pathlib.Path(os.path.join(args.wav_data, '/test')).mkdir(parents=True, exist_ok=True)

	train_file_list = glob.glob(os.path.join(args.split_data, '/train')+'/**/*.wav', recursive = True)
	test_file_list = glob.glob(os.path.join(args.split_data, '/test')+'/**/*.wav', recursive = True)

	idx_train, idx_test = idx[:n_train], idx[n_train:]

	for ref_file in train_mat_files:
		class_id = ref_file.split('/')[-2]
		current_file = os.path.join(args.wav_data, class_id, os.path.basename(ref_file).split('.')[0]) + '.wav'
		renamed_file = os.path.join(args.wav_data, class_id, '/train/', os.path.basename(ref_file).split('.')[0]) + '.wav'
		copyfile(current_file, current_file)

	for ref_file in test_mat_files:
		class_id = ref_file.split('/')[-2]
		current_file = os.path.join(args.wav_data, class_id, os.path.basename(ref_file).split('.')[0]) + '.wav'
		renamed_file = os.path.join(args.wav_data, class_id, '/test/', os.path.basename(ref_file).split('.')[0]) + '.wav'
		copyfile(current_file, current_file)
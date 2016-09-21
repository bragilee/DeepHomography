import os
import cv2
import sys
import numpy as np

data_path = '/Users/bragi/Computer_Vision/DeepHomography/graffiti_homography'

h_data_path = '/Users/bragi/Computer_Vision/DeepHomography/graffiti_dh'

def removeHiddenfile(directory_list):
	if '.' in directory_list:
		directory_list.remove('.')
	if '..' in directory_list:
		directory_list.remove('.')
	if '.DS_Store' in directory_list:
		directory_list.remove('.DS_Store')
	return directory_list

directory_list = removeHiddenfile(os.listdir(data_path))	

for directory in directory_list:
	# print(directory)
	directory_path = os.path.join(data_path, directory)
	file_list = removeHiddenfile(os.listdir(directory_path))
	h_directory_path = os.path.join(h_data_path, directory)
	if not os.path.exists(h_directory_path):
		os.mkdir(h_directory_path)

	for h_file in file_list:
		file_path = os.path.join(directory_path, h_file)
		
		f = open(file_path)
		data = f.read()		
		print(data)
		print(data.split('\t'))
		break
	break
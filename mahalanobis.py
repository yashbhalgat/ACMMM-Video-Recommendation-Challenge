import csv
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
import pdb
import pickle	# Use hdf5

import torch
import torch.nn as nn
from torch.autograd import Variable as var

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

def load_gt(shows_dir, movies_dir):
	## Shows gt 
	shows_train_gt = []
	fp = open(shows_dir+"relevance_train.csv", "r")
	for row in csv.reader(fp):
		shows_train_gt.append(row)
	fp.close()

	shows_val_gt = []
	fp = open(shows_dir+"relevance_val.csv", "r")
	for row in csv.reader(fp):
		shows_val_gt.append(row)
	fp.close()

	## Movies gt 
	movies_train_gt = []
	fp = open(movies_dir+"relevance_train.csv", "r")
	for row in csv.reader(fp):
		movies_train_gt.append(row)
	fp.close()

	movies_val_gt = []
	fp = open(movies_dir+"relevance_val.csv", "r")
	for row in csv.reader(fp):
		movies_val_gt.append(row)
	fp.close()

	return shows_train_gt, shows_val_gt, movies_train_gt, movies_val_gt

def load_set(shows_dir, movies_dir, phase="test"):
	# loading test set
	shows_set = genfromtxt(shows_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
	shows_set = list(shows_set)
	shows_set = [int(i) for i in shows_set]
	movies_set = genfromtxt(movies_dir+"split/"+phase+".csv", delimiter=',', dtype=str)
	movies_set = list(movies_set)
	movies_set = [int(i) for i in movies_set]

	return shows_set, movies_set

def build_inverted_index(train_gt):
	inv_index = {}
	for t in range(len(train_gt)):
		for rank, c in enumerate(train_gt[t][1:]):
			try:
				inv_index[int(c)].append((int(train_gt[t][0]),float(rank+1)))
			except KeyError:
				inv_index[int(c)] = []
				inv_index[int(c)].append((int(train_gt[t][0]),float(rank+1)))
			except ValueError:
				pass

	return inv_index



class Net(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		return out



if __name__ == '__main__':
	shows_train_gt, shows_val_gt, movies_train_gt, movies_val_gt = load_gt(shows_dir, movies_dir)
	shows_train_set, movies_train_set = load_set(shows_dir, movies_dir, "train")
	shows_valid_set, movies_valid_set = load_set(shows_dir, movies_dir, "val")
	shows_test_set, movies_test_set = load_set(shows_dir, movies_dir, "test")

	shows_set = [str(i) for i in range(7536)]
	movies_set = [str(i) for i in range(10826)]

	#### Load Features
	print("#"*50)
	print("=> Loading Features")
	reload_features = True

	if reload_features == False:
		shows_features = {}
		movies_features = {}

		for show in shows_set:
			folder = shows_dir+"feature/"
			c3d = np.load(folder+show+"/"+show+"-c3d-pool5.npy")
			inception = np.load(folder+show+"/"+show+"-inception-pool3.npy")
			inception = np.mean(inception,0)
			feature = np.concatenate((c3d, inception))
			shows_features[int(show)] = feature

		print("=> Loaded Shows Features")

		for movie in movies_set:
			folder = movies_dir+"feature/"
			c3d = np.load(folder+movie+"/"+movie+"-c3d-pool5.npy")
			inception = np.load(folder+movie+"/"+movie+"-inception-pool3.npy")
			inception = np.mean(inception,0)
			feature = np.concatenate((c3d, inception))
			movies_features[int(movie)] = feature

		print("=> Loaded Movies Features")

	else:
		shows_features = pickle.load(open("shows_features.pkl", "rb"))
		movies_features = pickle.load(open("movies_features.pkl", "rb"))

	print("=> Loading Features Done")


	# pdb.set_trace()

	## Checking dot products for shows_train_gt[0][0]
	# r = shows_features[int(shows_train_gt[0][0])]
	# for c in shows_train_gt[0][1:]:
	# 	vec = shows_features[int(c)]
	# 	similarity = np.dot(r, vec)/(norm(r)*norm(vec))
	# 	print(c, similarity)

	#####################################################################################
	shows_inv_ind = build_inverted_index(shows_train_gt)

	# pdb.set_trace()

	## Train using Neural Network
	input_size = 2560
	hidden_size = 1024
	output_size = 256
	num_epochs = 16

	net = Net(input_size, hidden_size, output_size)
	# net.cuda()

	criterion = torch.nn.MSELoss(size_average = False)
	optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
	
	for epoch in range(num_epochs):
		print("=> Start training neural network")
		for c in shows_inv_ind.keys():
			fc = shows_features[c]
			var_fc = var(torch.from_numpy(fc).float())#.cuda()
			embed_fc = net(var_fc)

			pred = 0
			for tr in shows_inv_ind[c]:
				var_ft = var(torch.from_numpy(shows_features[tr[0]]).float())#.cuda()
				pred += net(var_ft)*(1/tr[1])

			pred = pred/len(shows_inv_ind[c])

			target = var(torch.zeros(embed_fc.size()))

			loss = criterion(pred-embed_fc, target)

			# Zero gradients, perform a backward pass, 
			# and update the weights.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('epoch {}, loss {}'.format(epoch, loss.data[0]))


	pdb.set_trace()
	aaaa = 4
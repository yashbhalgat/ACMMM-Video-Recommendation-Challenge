import csv
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
import pdb
import pickle	# Use hdf5
# from mahalanobis import load_gt, load_set
import random

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

def load_gt(shows_dir, movies_dir):
	## Shows gt 
	shows_train_gt = []
	fp = open(shows_dir+"relevance_train.csv", "r")
	for row in csv.reader(fp):
		row = [i for i in row if i.isdigit()]
		shows_train_gt.append(row)
	fp.close()

	shows_val_gt = []
	fp = open(shows_dir+"relevance_val.csv", "r")
	for row in csv.reader(fp):
		row = [i for i in row if i.isdigit()]
		shows_val_gt.append(row)
	fp.close()

	## Movies gt 
	movies_train_gt = []
	fp = open(movies_dir+"relevance_train.csv", "r")
	for row in csv.reader(fp):
		row = [i for i in row if i.isdigit()]
		movies_train_gt.append(row)
	fp.close()

	movies_val_gt = []
	fp = open(movies_dir+"relevance_val.csv", "r")
	for row in csv.reader(fp):
		row = [i for i in row if i.isdigit()]
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

def get_features(shows_set, movies_set, shows_dir, movies_dir, feature_type="concat", reload_features=True):
	if reload_features == False:
		shows_features = {}
		movies_features = {}

		for show in shows_set:
			folder = shows_dir+"feature/"
			c3d = np.load(folder+show+"/"+show+"-c3d-pool5.npy")
			if feature_type=="concat":
				inception = np.load(folder+show+"/"+show+"-inception-pool3.npy")
				inception = np.mean(inception,0)
				feature = np.concatenate((c3d, inception))
			elif feature_type=="c3d":
				feature = c3d
			shows_features[int(show)] = feature

		print("=> Loaded Shows Features")

		for movie in movies_set:
			folder = movies_dir+"feature/"
			c3d = np.load(folder+movie+"/"+movie+"-c3d-pool5.npy")
			if feature_type=="concat":
				inception = np.load(folder+movie+"/"+movie+"-inception-pool3.npy")
				inception = np.mean(inception,0)
				feature = np.concatenate((c3d, inception))
			elif feature_type=="c3d":
				feature = c3d
			movies_features[int(movie)] = feature

		print("=> Loaded Movies Features")

		if feature_type=="concat":
			with open("shows_features.pkl", "wb") as fp:
				pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
			with open("movies_features.pkl", "wb") as fp:
				pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			with open("shows_features_"+feature_type+".pkl", "wb") as fp:
				pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
			with open("movies_features_"+feature_type+".pkl", "wb") as fp:
				pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)

		print("=> Saved Features")

	else:
		# Load features from saved pickle files
		if feature_type=="concat":
			shows_features = pickle.load(open("shows_features.pkl", "rb"))
			movies_features = pickle.load(open("movies_features.pkl", "rb"))
		else:
			shows_features = pickle.load(open("shows_features_"+feature_type+".pkl", "rb"))
			movies_features = pickle.load(open("movies_features_"+feature_type+".pkl", "rb"))


	print("=> Loading Features Done")
	return shows_features, movies_features


def get_norm_params(features):
	'''
	Usage: features = shows_features
	'''
	feat_length = features[0].shape[0]
	num_samples = len(features.keys())

	mean_vec = np.zeros((feat_length))
	for ind in range(feat_length):
		sum = 0
		for c in features.keys():
			sum += features[c][ind]
		mean_vec[ind] = sum/num_samples

	std_vec = np.zeros((feat_length))
	for ind in range(feat_length):
		std_vec[ind] = np.std([features[c][ind] for c in features.keys()])

	params = {}
	params["mean_vec"] = mean_vec
	params["std_vec"] = std_vec

	return params

def normalize_features(features, params):	# features = shows_features
	mean_vec = params["mean_vec"]
	std_vec = params["std_vec"]

	for c in features.keys():
		features[c] = (features[c]-mean_vec)/std_vec

	return features

def make_psd(W):
	""" Make matrix positive semi-definite. """
	w, v = np.linalg.eig(0.5 * (W + W.T))  # eigvec in columns
	D = np.diagflat(np.maximum(w, 0))
	W[:] = np.dot(np.dot(v, D), v.T)

def symmetrize(W):
	""" Symmetrize matrix. """
	W[:] = 0.5 * (W + W.T)

def bilinear_similarity(W, x1, x2):
	return np.dot(np.dot(x1, W), x2)

def perform_val(W, shows_features, shows_valid_set):
	'''Validation'''
	fp = open('track_1_shows/predict_val_my_features.csv','w')
	writer = csv.writer(fp)

	for c in shows_features.keys():
		shows_features[c] = shows_features[c]/norm(shows_features[c])

	for r in shows_valid_set[0:10]:
		sr = [bilinear_similarity(W, shows_features[r], shows_features[i]) for i in range(7356)]
		sr_sorted = sorted(sr, reverse=True)[0:500]		# sorted similarity values
		ind = [sr.index(i) for i in sr_sorted]
		ind = [r] + ind
		print(r, "done")
		writer.writerow(ind)

def kernel(x1, x2, kernel_type="gaussian", gamma=50, sigma=2):
	if kernel_type=="gaussian":
		d = norm(x1-x2)
		return np.exp(-d/(gamma*sigma**2))
	elif kernel_type=="dotprod":	# shifted inner product
		return 0.5*np.dot(x1,x2)/(norm(x1)*norm(x2))+0.5

def kernel_similarity_matrix(p_vid, q_vid, K, triplets, tau):
	T = len(triplets)
	score = K[p_vid, q_vid]
	# for t in range(T):
	# 	pl = triplets[t][0]
	# 	pl_plus = triplets[t][1]
	# 	pl_minus = triplets[t][2]
	# 	score += tau[t]*K[p_vid, pl]*( K[pl_plus,q_vid] - K[pl_minus,q_vid] )
	pl_inds = [triplets[t][0] for t in range(T)]
	plplus_inds = [triplets[t][1] for t in range(T)]
	plminus_inds = [triplets[t][2] for t in range(T)]
	
	K1 = K[p_vid, pl_inds]
	TAU = np.diag(tau)
	K2 = K[plplus_inds, q_vid] - K[plminus_inds, q_vid]

	score += np.dot(np.dot(K1, TAU), K2)

	return score

def kernel_similarity(p_vid, q_vid, K, triplets, tau):
	T = len(triplets)
	score = K[p_vid, q_vid]
	for t in range(T):
		pl = triplets[t][0]
		pl_plus = triplets[t][1]
		pl_minus = triplets[t][2]
		score += tau[t]*K[p_vid, pl]*( K[pl_plus,q_vid] - K[pl_minus,q_vid] )
	
	return score

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
	reload_features = False

	shows_features, movies_features = get_features(shows_set, movies_set, \
		shows_dir, movies_dir, feature_type="c3d", reload_features=reload_features)

	pdb.set_trace()

	####################################################################################
	# Kernelised OASIS Algorithm
	####################################################################################

	## NORMALIZE Features
	# for c in shows_features.keys():
	# 	norm = np.linalg.norm(shows_features[c])
	# 	if norm != 0:
	# 		shows_features[c] = shows_features[c]/norm
	params = get_norm_params(shows_features)
	shows_features = normalize_features(shows_features, params)

	## Building the Kernel matrix
	gamma = 67.14
	sigma = 2
	num_samples = len(shows_features.keys())
	
	# pdb.set_trace()

	generate_K = True
	print("Kernel being calculated")
	if generate_K:
		K = np.zeros((num_samples, num_samples))
		for i in range(num_samples):
			for j in range(num_samples):
				K[i,j] = kernel(shows_features[i], shows_features[j], gamma=gamma, sigma=sigma)
		np.save("kernel_c3d_normalised.npy", K)

	else:
		K = np.load("kernel_c3d_normalised.npy")

	print("Kernel calculated... Now saving kernel")

	print("Kernel saved... Now calculating triplets")
	
	triplets = []
	tau = []

	n_features = shows_features[0].shape[0]
	n_epochs = 1
	# n_iters = 10000

	train_size = len(shows_train_gt)

	aggress = 0.8

	## Generate triplets
	state = np.random.RandomState(None)
	print("PHASE 1")
	for iteration in range(1400):
		try:
			p_ind = state.randint(train_size)

			n_rel = len(shows_train_gt[p_ind])-1
			if n_rel < 2:
				continue

			pos_ind = state.randint(n_rel-1)+1
			neg_ind = state.randint(n_rel-pos_ind)+pos_ind+1

			p_vid = int(shows_train_gt[p_ind][0])
			pos_vid = int(shows_train_gt[p_ind][pos_ind])
			neg_vid = int(shows_train_gt[p_ind][neg_ind])

			# print(p_vid, pos_vid, neg_vid)
			triplets.append((p_vid, pos_vid, neg_vid))
		except:
			pass

	print("PHASE 2")
	for t in range(len(shows_train_gt)):
		try:
			rel_vids = [int(r) for r in shows_train_gt[t]]
			if len(rel_vids) < 2:
				continue

			for sub_iter in range(8):
				p_ind = state.randint(len(rel_vids))
				pos_ind = state.randint(len(rel_vids))
				while pos_ind==p_ind:
					pos_ind = state.randint(len(rel_vids))

				p_vid = rel_vids[p_ind]
				pos_vid = rel_vids[pos_ind]

				neg_vid = state.randint(len(shows_set))
				while neg_vid in rel_vids:
					neg_vid = state.randint(len(shows_set))

				# print(p_vid, pos_vid, neg_vid)
				triplets.append((p_vid, pos_vid, neg_vid))
		except:
			pass

	triplets = triplets*8
	random.shuffle(triplets)

	np.save("triplets.npy", triplets)

	for epoch in range(n_epochs):
		print("#"*50)
		print("\n ===> EPOCH NO. ", epoch)

		for t, trio in enumerate(triplets):
			print(t, trio)
			p = trio[0]
			p_plus = trio[1]
			p_minus = trio[2]

			Sp = K[p, p_plus]
			Sn = K[p, p_minus]

			for l in range(t):
				pl = triplets[l][0]
				pl_plus = triplets[l][1]
				pl_minus = triplets[l][2]
				taul = tau[l]

				Sp = Sp + taul*K[p, pl]*(K[pl_plus, p_plus]-K[pl_minus, p_plus])
				Sn = Sn + taul*K[p, pl]*(K[pl_plus, p_minus]-K[pl_minus, p_minus])

			tau_temp = max(0,1-Sp+Sn)/( K[p,p] * ( K[p_plus,p_plus]+K[p_minus,p_minus]-2*K[p_plus,p_minus] ))
			tau_curr = min(aggress, tau_temp)

			print("tau_temp", tau_temp)
			print("tau", tau_curr)
			tau.append(tau_curr)

	np.save("tau.npy", tau)

	pdb.set_trace()
	aaaa = 4

	# perform_val(W, shows_features, shows_valid_set)


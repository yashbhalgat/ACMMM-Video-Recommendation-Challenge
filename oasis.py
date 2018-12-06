import csv
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
import pdb
import pickle	# Use hdf5
from mahalanobis import load_gt, load_set

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

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

def bilinear_similarity(W, x1, x2):
	return np.dot(np.dot(x1, W), x2)/(norm(x1)*norm(x2))

def cosine_similarity(x1, x2):
	return np.dot(x1,x2)/(norm(x1)*norm(x2))

def gaussian_similarity(x1, x2, gamma=0.67, sigma=2):
	return np.exp(-np.linalg.norm(x1-x2)**2/((gamma**2)*(sigma**2)))

def make_psd(W):
	""" Make matrix positive semi-definite. """
	w, v = np.linalg.eig(0.5 * (W + W.T))  # eigvec in columns
	D = np.diagflat(np.maximum(w, 0))
	W[:] = np.dot(np.dot(v, D), v.T)

def symmetrize(W):
	""" Symmetrize matrix. """
	W[:] = 0.5 * (W + W.T)

def perform_val(W, shows_features, shows_valid_set):
	'''Validation'''
	fp = open('track_1_shows/predict_val_my_features.csv','w')
	writer = csv.writer(fp)

	for c in shows_features.keys():
		shows_features[c] = shows_features[c]/norm(shows_features[c])

	for r in shows_valid_set:
		sr = [bilinear_similarity(W, shows_features[r], shows_features[i]) for i in range(7356)]
		sr_sorted = sorted(sr, reverse=True)[0:500]		# sorted similarity values
		ind = [sr.index(i) for i in sr_sorted]
		ind = [r] + ind
		writer.writerow(ind)


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

	####################################################################################
	# OASIS Algorithm

	#####################################################
	### NORMALIZE Features first???
	for c in shows_features.keys():
		norm = np.linalg.norm(shows_features[c])
		if norm != 0:
			shows_features[c] = shows_features[c]/norm
	#####################################################

	n_features = shows_features[0].shape[0]
	n_epochs = 10
	# n_iters = 10000

	train_size = len(shows_train_gt)

	aggress = 0.1
	W = np.eye(n_features)

	for epoch in range(n_epochs):
		print("#"*50)
		print("\n ===> EPOCH NO. ", epoch)

		state = np.random.RandomState(None)
		count_iter = 0

		### Sampling triplets with p- from relevance list
		for iteration in range(300):
			p_ind = state.randint(train_size)

			n_rel = len(shows_train_gt[p_ind])-1
			if n_rel < 2:
				continue

			pos_ind = state.randint(n_rel-1)+1
			neg_ind = state.randint(n_rel-pos_ind)+pos_ind+1

			p_vid = int(shows_train_gt[p_ind][0])
			pos_vid = int(shows_train_gt[p_ind][pos_ind])
			neg_vid = int(shows_train_gt[p_ind][neg_ind])

			p = shows_features[p_vid]
			samples_delta = shows_features[pos_vid]-shows_features[neg_vid]

			loss = 1 - np.dot(np.dot(p, W), samples_delta)

			if loss > 0:
				# Update W
				grad_W = np.outer(p, samples_delta)

				norm_grad_W = np.dot(p, p) * np.dot(samples_delta,
													samples_delta)

				# constraint on the maximal update step size
				tau_val = loss / norm_grad_W  # loss / (V*V.T)
				tau = np.minimum(aggress, tau_val)

				W += tau * grad_W

			count_iter += 1
			if (count_iter)%100 == 0:
				make_psd(W)
				symmetrize(W)

			if (count_iter)%1000 == 0:
				np.save("W_"+str(count_iter)+".npy", W)

			print("triplets: (", p_vid, ",", pos_vid, ",", neg_vid, ") loss:", loss)

		print("#"*50)
		print("\n PHASE 1 of epoch", epoch, "DONE")
		print("#"*50)

		### Sampling triplets with p- from outside the relevance list
		for t in range(len(shows_train_gt)):
			rel_vids = [int(r) for r in shows_train_gt[t]]

			for sub_iter in range(1):
				p_ind = state.randint(len(rel_vids))
				pos_ind = state.randint(len(rel_vids))
				p_vid = rel_vids[p_ind]
				pos_vid = rel_vids[pos_ind]

				neg_vid = state.randint(len(shows_set))
				while neg_vid in rel_vids:
					neg_vid = state.randint(len(shows_set))

				p = shows_features[p_ind]
				samples_delta = shows_features[pos_ind]-shows_features[neg_ind]

				loss = 1 - np.dot(np.dot(p, W), samples_delta)

				if loss > 0:
					# Update W
					grad_W = np.outer(p, samples_delta)

					norm_grad_W = np.dot(p, p) * np.dot(samples_delta,
														samples_delta)

					# constraint on the maximal update step size
					tau_val = loss / norm_grad_W  # loss / (V*V.T)
					tau = np.minimum(aggress, tau_val)

					W += tau * grad_W

				count_iter += 1
				if (count_iter)%100 == 0:
					make_psd(W)
					symmetrize(W)

				if (count_iter)%1000 == 0:
					np.save("W_"+str(count_iter)+".npy", W)

				print("triplets: (", p_vid, ",", pos_vid, ",", neg_vid, ") loss:", loss)

		print("#"*50)
		print("\n PHASE 2 of epoch", epoch, "DONE")
		print("#"*50)

	make_psd(W)
	symmetrize(W)

	pdb.set_trace()
	aaaa = 4

	perform_val(W, shows_features, shows_valid_set)


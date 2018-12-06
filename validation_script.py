import csv
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
import pdb
import pickle	# Use hdf5
from oasis import load_set, get_norm_params, normalize_features, bilinear_similarity, cosine_similarity, gaussian_similarity

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

def find_all(my_list, val):
	indices = [i for i, x in enumerate(my_list) if x == val]
	return indices

def perform_val(W, shows_features, shows_valid_set, num_iterations):
	'''Validation'''
	fp = open('track_1_shows/predict_val_bilinear_normalised_unitnorm_'+str(num_iterations)+'.csv','w')
	writer = csv.writer(fp)

	params = get_norm_params(shows_features)
	shows_features = normalize_features(shows_features, params)

	## For unitnorm models
	for c in shows_features.keys():
		shows_features[c] = shows_features[c]/norm(shows_features[c])

	for r in shows_valid_set[0:10]:
		sr = [bilinear_similarity(W, shows_features[r], shows_features[i]) for i in range(7356)]
		sr_sorted = sorted(sr, reverse=True)[0:500]		# sorted similarity values
		ind = [sr.index(i) for i in sr_sorted]
		ind = [r] + ind
		print(r, "done")
		writer.writerow(ind)

def perform_val_cosine(shows_features, shows_valid_set):
	'''Validation with cosine similarity'''
	fp = open('track_1_shows/predict_val_cosine_unitnorm_normalised.csv','w')
	writer = csv.writer(fp)

	for c in shows_features.keys():
		shows_features[c] = shows_features[c]/norm(shows_features[c])

	params = get_norm_params(shows_features)
	shows_features = normalize_features(shows_features, params)

	for r in shows_valid_set:
		sr = [cosine_similarity(shows_features[r], shows_features[i]) for i in range(7356)]
		sr_sorted = sorted(sr, reverse=True)[0:500]		# sorted similarity values
		ind = [sr.index(i) for i in sr_sorted]
		ind = [r] + ind
		print(r, "done")
		writer.writerow(ind)

def perform_val_kernel(shows_valid_set, T):
	'''Validation'''
	fp = open('track_1_shows/predict_val_kernel_combined_'+str(T)+'.csv','w')
	writer = csv.writer(fp)

	scores1 = np.load("kernel_stats/score_kernel_stats_"+str(T)+".npy")
	scores2 = np.load("kernel_LSTM_out/score_kernel_rbf_LSTM_smaller_embedd200_54070.npy")
	scores = scores1 + scores2

	for num, r in enumerate(shows_valid_set):
		#pdb.set_trace()
		sr = list(scores[num])

		ind_all = sorted(range(len(sr)), reverse=True, key=lambda k: sr[k])
		ind = ind_all[0:501]
		if r in ind:
			ind.remove(r)

		# ind = [sr.index(i) for i in sr_sorted]
		ind = [r] + ind
		print(r, "done")
		writer.writerow(ind)


def perform_val_cosine_deeplearning(shows_valid_set, model_type):
	'''Validation with cosine similarity'''
	fp = open('track_1_shows/predict_val_'+model_type+'.csv','w')
	writer = csv.writer(fp)

	shows_embeddings = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+model_type+".pkl", "rb"))

	# params = get_norm_params(shows_embeddings)
	# shows_embeddings = normalize_features(shows_embeddings, params)

	for r in shows_valid_set:
		sr = [cosine_similarity(shows_embeddings[r][0,:], shows_embeddings[i][0,:]) for i in range(7356)]
		ind_all = sorted(range(len(sr)), reverse=True, key=lambda k: sr[k])
		ind = ind_all[0:501]
		if r in ind:
			ind.remove(r)
		ind = [r] + ind
		print(r, "done")
		writer.writerow(ind)

def perform_val_exp_deeplearning(shows_valid_set, model_type):
	'''Validation with cosine similarity'''
	fp = open('track_1_shows/predict_val_'+model_type+'.csv','w')
	writer = csv.writer(fp)

	shows_embeddings = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+model_type+".pkl", "rb"))

	# params = get_norm_params(shows_embeddings)
	# shows_embeddings = normalize_features(shows_embeddings, params)
	
	for r in shows_valid_set:
		sr = [np.exp(-norm(shows_embeddings[r]-shows_embeddings[i])) for i in range(7356)]

		ind_all = sorted(range(len(sr)), reverse=True, key=lambda k: sr[k])
		ind = ind_all[0:501]
		if r in ind:
			ind.remove(r)
		ind = [r] + ind
		# print(r, "done")
		writer.writerow(ind)

def perform_val_exp_deeplearning_intersection(shows_valid_set):
	'''Validation with cosine similarity'''
	fp = open('track_1_shows/predict_val_'+model_type+'.csv','w')
	writer = csv.writer(fp)

	shows_embeddings1 = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+"LSTMNet_smaller_softmax_margin035_embedd200_1500001"+".pkl", "rb"))
	shows_embeddings2 = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+"LSTMNet_small_vary_softmax_margin035_embedd40_1400001"+".pkl", "rb"))
	# shows_embeddings3 = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+model_type+".pkl", "rb"))
	# shows_embeddings4 = pickle.load(open("deeplearning/LSTMNet_out/shows_embedd_"+model_type+".pkl", "rb"))
	
	# params = get_norm_params(shows_embeddings)
	# shows_embeddings = normalize_features(shows_embeddings, params)
	
	for r in shows_valid_set:
		sr = [np.exp(-norm(shows_embeddings[r]-shows_embeddings[i])) for i in range(7356)]

		ind_all = sorted(range(len(sr)), reverse=True, key=lambda k: sr[k])
		ind = ind_all[0:501]
		if r in ind:
			ind.remove(r)
		ind = [r] + ind
		# print(r, "done")
		writer.writerow(ind)


if __name__ == '__main__':
	# num_iterations = 40000
	# W = np.load("Bilinear_Unitnorm_models/W_"+str(num_iterations)+".npy")
	shows_features = pickle.load(open("shows_features.pkl", "rb"))
	shows_valid_set, movies_valid_set = load_set(shows_dir, movies_dir, "val")
	shows_train_set, movies_valid_set = load_set(shows_dir, movies_dir, "train")

	# perform_val(W, shows_features, shows_valid_set, num_iterations)
	# perform_val_cosine(shows_features, shows_valid_set)

	# K = np.load("kernel_LSTM_out/kernel_LSTM_smaller_embedd200.npy")
	# triplets = np.load("kernel_LSTM_out/triplets.npy")
	# tau = np.load("kernel_LSTM_out/tau.npy")
	perform_val_kernel(shows_valid_set, 30501)

	# perform_val_cosine_deeplearning(shows_valid_set)
	
	### Performing tests on LSTM models
	# models = ["LSTMNet_small_vary_softmax_margin035_embedd40"]
	# # epochs = [100001, 200001, 300001, 400001, 500001, 600001, 700001, 800001, 900001, 1000001,\
	# # 			1100001, 1200001, 1300001, 1400001, 1500001]
	# epochs = [1400001]

	# for model in models:
	# 	for epoch in epochs:
	# 		print(model+"_"+str(epoch))
	# 		perform_val_exp_deeplearning(shows_valid_set, model+"_"+str(epoch))

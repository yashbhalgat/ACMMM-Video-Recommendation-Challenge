import pickle
import argparse
import numpy as np
import pandas as pd
import pdb

def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

def get_features(shows_set, movies_set, shows_dir, movies_dir, feature_type="concat", reload_features=True):
	if reload_features == False:
		shows_features = {}
		movies_features = {}

		shows_inception = pickle.load(open("shows_stats_delta_features.pkl", "rb"))
		movies_inception = pickle.load(open("movies_stats_delta_features.pkl", "rb"))

		for show in shows_set:
			folder = shows_dir+"feature/"
			c3d = np.load(folder+show+"/"+show+"-c3d-pool5.npy")
			if feature_type=="concat":
				# inception = np.load(folder+show+"/"+show+"-inception-pool3.npy")
				# inception = np.mean(inception,0)
				inception = shows_inception[int(show)]
				feature = np.concatenate((c3d, inception))
			elif feature_type=="c3d":
				feature = c3d
			shows_features[int(show)] = feature

		print("=> Loaded Shows Features")

		for movie in movies_set:
			folder = movies_dir+"feature/"
			c3d = np.load(folder+movie+"/"+movie+"-c3d-pool5.npy")
			if feature_type=="concat":
				# inception = np.load(folder+movie+"/"+movie+"-inception-pool3.npy")
				# inception = np.mean(inception,0)
				inception = movies_inception[int(movie)]
				feature = np.concatenate((c3d, inception))
			elif feature_type=="c3d":
				feature = c3d
			movies_features[int(movie)] = feature

		print("=> Loaded Movies Features")

		# if feature_type=="concat":
		# 	with open("shows_features.pkl", "wb") as fp:
		# 		pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# 	with open("movies_features.pkl", "wb") as fp:
		# 		pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# else:
		# 	with open("shows_features_"+feature_type+".pkl", "wb") as fp:
		# 		pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# 	with open("movies_features_"+feature_type+".pkl", "wb") as fp:
		# 		pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)

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


def get_video_features(shows_set, shows_dir, movies_set=[], movies_dir=None, reload_features=False):
	if reload_features == False:
		shows_features = {}
		movies_features = {}

		print("=> Loading features started!")

		for show in shows_set:
			# print(show)
			folder = shows_dir+"feature/"
			inception = np.load(folder+show+"/"+show+"-inception-pool3.npy")
			shows_features[int(show)] = inception
			# shows_features[int(show)] = np.mean(inception.reshape(inception.shape[0], 64, -1), 2)

		print("=> Loaded Shows Features")

		for movie in movies_set:
			# print(movie)
			folder = movies_dir+"feature/"
			inception = np.load(folder+movie+"/"+movie+"-inception-pool3.npy")
			movies_features[int(movie)] = inception
			# movies_features[int(movie)] = np.mean(inception.reshape(inception.shape[0], 64, -1), 2)

		print("=> Loaded Movies Features")

		# with open("shows_video_features.pkl", "wb") as fp:
		# 	pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# with open("movies_video_features.pkl", "wb") as fp:
		# 	pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)

	return shows_features, movies_features

def get_stats_features(shows_set=[], shows_dir=None, movies_set=[], movies_dir=None, reload_features=False):
	if reload_features == False:
		shows_features = {}
		movies_features = {}

		print("=> Loading features started!")

		shows_stats = pickle.load(open("shows_stats_features.pkl", "rb"))
		movies_stats = pickle.load(open("movies_stats_features.pkl", "rb"))

		for show in shows_set:
			print(show)
			folder = shows_dir+"feature/"
			inception = np.load(folder+show+"/"+show+"-inception-pool3.npy")
			rows, cols = 64, 32
			window_size1, window_size2 = 4, 2
			n_frames = inception.shape[0]
			mean_inception = inception.reshape(n_frames, rows//window_size1, window_size1, cols//window_size2, window_size2).mean(axis=(2, 4))
			
			inception = mean_inception.reshape(mean_inception.shape[0], -1)
			delta_feat = delta(inception, 1)
			delta_delta_feat = delta(inception, 2)

			stats_feature = shows_stats[int(show)]
			# pdb.set_trace()
			for i in range(inception.shape[1]):
				stats_feature = np.append(stats_feature, np.asarray(pd.Series(delta_feat[:,i]).describe()[1:]))
			# pdb.set_trace()
			for i in range(inception.shape[1]):
				stats_feature = np.append(stats_feature, np.asarray(pd.Series(delta_delta_feat[:,i]).describe()[1:]))
			# pdb.set_trace()
			shows_features[int(show)] = stats_feature
			# shows_features[int(show)] = np.mean(inception.reshape(inception.shape[0], 64, -1), 2)

		print("=> Loaded Shows Features")

		for movie in movies_set:
			print(movie)
			folder = movies_dir+"feature/"
			inception = np.load(folder+movie+"/"+movie+"-inception-pool3.npy")
			rows, cols = 64, 32
			window_size1, window_size2 = 4, 2
			n_frames = inception.shape[0]
			mean_inception = inception.reshape(n_frames, rows//window_size1, window_size1, cols//window_size2, window_size2).mean(axis=(2, 4))
			
			inception = mean_inception.reshape(mean_inception.shape[0], -1)
			delta_feat = delta(inception, 1)
			delta_delta_feat = delta(inception, 2)

			stats_feature = movies_stats[int(movie)]
			# pdb.set_trace()
			for i in range(inception.shape[1]):
				stats_feature = np.append(stats_feature, np.asarray(pd.Series(delta_feat[:,i]).describe()[1:]))
			# pdb.set_trace()
			for i in range(inception.shape[1]):
				stats_feature = np.append(stats_feature, np.asarray(pd.Series(delta_delta_feat[:,i]).describe()[1:]))
			# pdb.set_trace()

			movies_features[int(movie)] = stats_feature
			# movies_features[int(movie)] = np.mean(inception.reshape(inception.shape[0], 64, -1), 2)

		print("=> Loaded Movies Features")

		with open("shows_stats_delta_features.pkl", "wb") as fp:
			pickle.dump(shows_features, fp, protocol=pickle.HIGHEST_PROTOCOL)
		# with open("movies_stats_delta_features.pkl", "wb") as fp:
		# 	pickle.dump(movies_features, fp, protocol=pickle.HIGHEST_PROTOCOL)

	return shows_features, movies_features


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Stats features')
	parser.add_argument('--type', default='shows', type=str,
                    help='name of similarity measure')
	args = parser.parse_args()

	main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"
	shows_dir = main_dir + "track_1_shows/"
	movies_dir = main_dir + "track_2_movies/"
	shows_set = [str(i) for i in range(7536)]
	movies_set = [str(i) for i in range(10826)]
	get_stats_features(shows_set=shows_set, shows_dir=shows_dir, movies_set=[], movies_dir=None, reload_features=False)
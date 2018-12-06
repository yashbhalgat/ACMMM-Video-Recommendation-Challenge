import torch
from tripletnetwork import Tripletnet, Net, LSTMNet, FusedLSTMNet, SimpleNet
import pickle
from torch.autograd import Variable
from train import get_norm_params, normalize_features
import pdb

from utils import get_video_features, get_features

main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

if __name__ == '__main__':
	# model_type = "conv_exp_margin_03"
	# model_types = ["LSTMNet_softmax_margin04_embedd128", \
	# 			"LSTMNet_softmax_margin04_embedd256", \
	# 			"LSTMNet_softmax_margin04_embedd64",\
	# 			"LSTMNet_cosine_margin04_embedd256"]

	# model_types = ["FusedLSTM_cosine_margin1_embedd256", "FusedLSTM_distance_margin1_embedd256"]
	model_types = ["SimpleNet_margin1_delta_SHOWS"]

	epochs = [100001, 200001, 300001, 400001, 500001, 600001, 700001, 800001, 900001, 1000001, 1100001]

	shows_set = [str(i) for i in range(7536)]
	movies_set = []
	# shows_features, _ = get_video_features(shows_set, shows_dir, reload_features=False)
	# shows_c3d, _ = get_features(shows_set, movies_set, shows_dir, movies_dir, feature_type="c3d", reload_features=False)
	
	# For SimpleNet
	####################################################################
	shows_features, _ = get_features(shows_set, movies_set, shows_dir, movies_dir, feature_type="concat", reload_features=False)
	params = get_norm_params(shows_features)
	features = normalize_features(shows_features, params)
	####################################################################

	pdb.set_trace()

	for model_type in model_types:
		for epoch in epochs:
			# embeddsize = int(model_type.split("_")[-1][6:])
			
			# epoch += 1500000

			net = SimpleNet()
			model = Tripletnet(net)

			checkpoint = torch.load("runs/"+model_type+"/checkpoint_"+str(epoch)+".pth.tar")
			print(model_type+"/checkpoint_"+str(epoch))
			# pdb.set_trace()
			model.load_state_dict(checkpoint['state_dict'])

			if torch.cuda.is_available():
				model.cuda()

			shows_embeddings = {}

			for c in features.keys():
				print(c, end=" ")
				fc = torch.from_numpy(features[c]).float().unsqueeze(0)
				# fc_c3d = torch.from_numpy(shows_c3d[c]).float().unsqueeze(0)
				if torch.cuda.is_available():
					fc = fc.cuda()
					# fc_c3d = fc_c3d.cuda()
				fc = Variable(fc)
				# fc_c3d = Variable(fc_c3d)

				embedding = model.embeddingnet(fc)
				shows_embeddings[c] = embedding.cpu().data.numpy()

			with open("LSTMNet_out/shows_embedd_"+model_type+"_"+str(epoch)+".pkl", "wb") as fp:
				pickle.dump(shows_embeddings, fp, protocol=pickle.HIGHEST_PROTOCOL)

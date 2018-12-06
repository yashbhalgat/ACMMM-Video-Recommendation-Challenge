import pickle
from scipy.spatial.distance import euclidean
import csv
import numpy as np
from numpy import genfromtxt
from numpy.linalg import norm
import pdb
# from mahalanobis import load_gt, load_set
import matplotlib.pyplot as plt


main_dir = "/scratch/jiadeng_fluxoe/yashsb/ACMMM_challenge/release/release/"

shows_dir = main_dir + "track_1_shows/"
movies_dir = main_dir + "track_2_movies/"

movies_features = pickle.load(open("movies_features.pkl", "rb"))
shows_features = pickle.load(open("shows_embeddings.pkl", "rb"))

distances_3 = [euclidean(shows_features[3], shows_features[i]) for i in range(4, 7356)]
# figure()
# plt.hist(distances_3, normed=False, bins=300)
# plt.savefig("distances_3.png")

# shows_train_gt, shows_val_gt, movies_train_gt, movies_val_gt = load_gt(shows_dir, movies_dir)
# shows_train_set, movies_train_set = load_set(shows_dir, movies_dir, "train")
# shows_valid_set, movies_valid_set = load_set(shows_dir, movies_dir, "val")
# shows_test_set, movies_test_set = load_set(shows_dir, movies_dir, "test")
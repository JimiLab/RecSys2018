
from sklearn.decomposition import TruncatedSVD
from predict import Predict
from DataManager import load_data
import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Z-score divide by zero is handled
from tqdm import tqdm
import os
from metrics import get_all_metrics
from scipy.sparse import lil_matrix
from scipy.stats import zscore
import random


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class PredictWithLSA(Predict):

    def __init__(self, data, num_components=64, lsa_min_track_prior=0.0):
        """

        :param data:
        :param num_components:
        :param lsa_min_track_prior: minimum proportion of playlists that a tracks has to appear in to be used for SVD
        projection. Default is 0.0, but a good baseline is 0.0002 or 0.02% of playlists. This retains about 22.5% of tracks.
        """
        # Call init on super class
        Predict.__init__(self)
        self.d = data  # DataManager Object
        self.num_components = num_components
        self.num_predictions = 500
        self.svd = TruncatedSVD(n_components=self.num_components)
        self.min_track_prior = lsa_min_track_prior

        self.lsa_track_mask = np.extract(self.d.popularity_vec > self.min_track_prior,
                                         np.arange(0,self.d.popularity_vec.shape[0]))
        self.lsa_id_to_column = dict()
        for i in range(len(self.lsa_track_mask)):
            self.lsa_id_to_column[self.lsa_track_mask[i]] = i

    def learn_model(self):

        if not hasattr(self.d, 'X'):
            print("Pickle File does not have pre-computed numpy X matrix. Aborting")
            return

        self.svd.fit(self.d.X[:, self.lsa_track_mask])

    def predict_playlists(self, lsa_weight= 0.0, popularity_weight=0.0, related_tracks_weight=0.0, z_score=False,
                          random_baseline=False):
        print("\nStarting playlist prediction...")

        num_subtest = len(self.d.test)
        num_playlists = len(self.d.test[0])
        metric_names = ["r_prec", "ndcg", "clicks"]
        num_metrics = len(metric_names)


        results = np.zeros((num_subtest,num_playlists, num_metrics), dtype=float)

        pbar = tqdm(total=num_subtest)
        pbar.write('~~~~~~~ Predicting Playlists ~~~~~~~')
        # INDUCED BUG - DO not start loop at 3rd subchallange
        for st in range(num_subtest):
        #for st in range(2,num_subtest):

            embedded_test_vecs = self.svd.transform(self.d.X_test[st][:, self.lsa_track_mask] )
            lsa_vecs_hat_compressed = self.svd.inverse_transform(embedded_test_vecs)

            if z_score:
                lsa_vecs_hat_compressed = zscore(lsa_vecs_hat_compressed, axis=1, ddof=1)
                np.nan_to_num(lsa_vecs_hat_compressed, copy=False)


            lsa_vecs_hat = lil_matrix(self.d.X_test[st].shape, dtype="float32")
            lsa_vecs_hat[:,self.lsa_track_mask] = lsa_vecs_hat_compressed

            #n, bins, patches = plt.hist(self.d.popularity_vec, 100, facecolor='blue', alpha=0.5)
            #plt.show()

            # linear combination of LSA score, popularity, and top tracks from artist and album
            test_vecs_hat = lsa_weight * lsa_vecs_hat + \
                            popularity_weight * self.d.popularity_vec + \
                            related_tracks_weight * self.d.X_test_top_tracks[st]

            # effectively remove known tracks that already appear in the test playlists by given large negative weight
            test_vecs_hat = test_vecs_hat -  self.d.X_test[st]* 99999999


            test_rank = np.argsort(-1 * test_vecs_hat, axis=1)

            if random_baseline:  # Change to True for Random Baseline
                np.random.shuffle(test_rank.T)

            test_rank = test_rank[:, 0:self.num_predictions]


            for pl in range(num_playlists):
                rank_list = test_rank[pl,:].tolist()[0]
                result = get_all_metrics(self.d.test_truth[st][pl], rank_list, self.num_predictions)
                results[st][pl] = np.array(result)

                    #  ignores test set songs not found in training set
            pbar.update(1)
        pbar.close()

        average_result = np.mean(results, axis=1)

        print("Weights (LSA, Pop, Related Track):", lsa_weight, popularity_weight, related_tracks_weight )
        print("Number Training Playlists and Tracks:", self.d.X.shape)
        print("LSA dims: ", self.num_components)
        print("LSA Track Corpus Size:", self.lsa_track_mask.size, "(min track prior =", self.min_track_prior,")")
        print()
        # print("Average Recall @ ", num_predictions,":", np.average(recall_500))
        self.print_subtest_results(self.d.subtest_name, metric_names, average_result)
        print()
        self.print_overall_results(metric_names, np.mean(average_result, axis=0))



    def generate_submission(self, filepath, popularity_weight=0.0):

        print("Encoding and Recoding Challenge Set Matrix")

        f = open(filepath, 'w')
        f.write("team_info,main,JimiLab,dougturnbull@gmail.com\n")

        X_challenge_embedded = self.svd.transform(self.d.X_challenge)
        X_challenge_hat = self.svd.inverse_transform(X_challenge_embedded)

        X_challenge_hat -= self.d.X_challenge * -99999999


        X_challenge_hat = (1 - popularity_weight) * X_challenge_hat + \
                          popularity_weight * self.d.popularity_vec[self.d.lsa_track_mask]

        rank = np.argsort(-1 * X_challenge_hat, axis=1)
        rank = rank[:, 0:self.num_predictions]

        pbar = tqdm(total=len(self.d.challenge))
        pbar.write('~~~~~~~ Generating Challenge Set Submission CSV File ~~~~~~~')

        for pid, playlist in d.challenge.items():

            spotify_pid = self.d.pid_to_spotify_pid[pid]
            f.write(str(spotify_pid))

            for tid in rank[pid]:
                f.write("," + str(self.d.id_to_uri[tid][0]))
            f.write("\n")
            pbar.update(1)

        pbar.close()
        f.close()


if __name__ == '__main__':

    """ Parameters for Loading Data """
    generate_data_arg = False   # True - load data for given parameter settings
    #                             False - only load data if pickle file doesn't already exist
    train_size_arg = 100000      # number of playlists for training
    test_size_arg = 1000        # number of playlists for testing
    load_challenge_arg = False   # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge dat (should always be True)
    random_baseline_arg = False  # set to true if you want to run random baseline

    num_components_arg = 128
    lsa_min_track_prior_arg = 0.0002  # minimum prior probability needed to keep track in LSA training matrix size (default 0.0002 or 2 / 10000 playlists
    lsa_zscore_arg = True             # zscore the output of the LSA weight after embedding and projecting back into the original space

    lsa_weight_arg = 1            # weight of LSA in linear combination
    popularity_weight_arg = 1     # set to 0 for no popularity bias, set to 1 for popularity baseline
    related_track_weight_arg = 1  # weight for top tracks from albums and artists already in the playlist




    submission_file_arg = os.path.join(os.getcwd(), 'data/submissions/lsa_test.csv')

    d = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg,
                  generate_data_arg)

    lsa = PredictWithLSA(d, num_components=num_components_arg, lsa_min_track_prior=lsa_min_track_prior_arg)
    lsa.learn_model()

    weights = [0, 0.0001, 0.001, 0.01, 0.1, .2, .5, .8, .9, 0.99, 0.999, 0.9999,1]
    for i in range(20):
        lsa_w = random.choice(weights)
        while (True):
            pop_w = random.choice(weights)
            if pop_w + lsa_w <= 1.0:
                break

        rt_w  = 1-lsa_w-pop_w

        lsa.predict_playlists(lsa_weight=lsa_w, popularity_weight=pop_w,related_tracks_weight=rt_w,
                              random_baseline=random_baseline_arg, z_score=True)

    #lsa.predict_playlists(popularity_weight=popularity_weight_arg, random_baseline=random_baseline_arg, z_score=True)


    if load_challenge_arg:
        print("Generating Submission file:", submission_file_arg)
        lsa.generate_submission(submission_file_arg, popularity_weight=popularity_weight_arg)
    print("done")


from sklearn.decomposition import TruncatedSVD
from predict import Predict
from DataManager import load_data
import math
import numpy as np
np.seterr(divide='ignore', invalid='ignore') # Z-score divide by zero is handled
from tqdm import tqdm
import os
from metrics import get_all_metrics
from scipy.sparse import lil_matrix, csc_matrix
from scipy.stats import zscore
import random
from sklearn.externals import joblib


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

        print("Learning LSA model...", end="")
        self.svd.fit(self.d.X[:, self.lsa_track_mask])
        print("done.")

    def predict_from_matrices(self, X, pop_vec, X_top_tracks, X_words, weights, z_score=True, random_baseline=False):


        #return np.random.randint(0, 10000 ,size=(X.shape[0], self.num_predictions))

        embedded_test_vecs = self.svd.transform(X[:, self.lsa_track_mask])
        lsa_vecs_hat_compressed = self.svd.inverse_transform(embedded_test_vecs)

        if z_score:
            lsa_vecs_hat_compressed = zscore(lsa_vecs_hat_compressed, axis=1, ddof=1)
            np.nan_to_num(lsa_vecs_hat_compressed, copy=False)

        lsa_vecs_hat = csc_matrix(X.shape, dtype="float32")
        lsa_vecs_hat[:, self.lsa_track_mask] = lsa_vecs_hat_compressed



        # linear combination of LSA score, popularity, and top tracks from artist and album
        test_vecs_hat = weights[0] * lsa_vecs_hat + \
                        weights[1] * pop_vec + \
                        weights[2] * X_top_tracks + \
                        weights[3] * X_words

        # effectively remove known tracks that already appear in the test playlists by given large negative weight
        test_vecs_hat = test_vecs_hat - X * 99999999

        test_rank = np.argsort(-1 * test_vecs_hat, axis=1)

        if random_baseline:  # Change to True for Random Baseline
            np.random.shuffle(test_rank.T)

        return test_rank[:, 0:self.num_predictions]

    def predict_from_words(self, mat):


        test_rank = np.argsort(-1 * mat.todense(), axis=1)

        return test_rank[:, 0:self.num_predictions]

    def predict_playlists(self, weights, z_score=False, random_baseline=False):

        """ weights = (lsa_weight, popularity_weight, related_tracks_weight, word_weight)"""

        #print("\nStarting playlist prediction...")
        print("Weights (LSA, Pop, Related Track, Title Words):", weights )


        num_subtest = len(self.d.test)
        num_playlists = len(self.d.test[0])
        metric_names = ["r_prec", "ndcg", "clicks"]
        num_metrics = len(metric_names)

        results = np.zeros((num_subtest,num_playlists, num_metrics), dtype=float)

        # create all popularity vecs so that 1st place is pop of 1.0
        pop_vec = self.d.popularity_vec / np.max(self.d.popularity_vec)

        pbar = tqdm(total=num_subtest)
        pbar.write('~~~~~~~ Predicting Playlists ~~~~~~~')

        for st in range(num_subtest):

            test_rank = self.predict_from_matrices(self.d.X_test[st].tocsc(),
                                                   pop_vec,
                                                   self.d.X_test_top_tracks[st],
                                                   self.d.X_test_words[st],
                                                   weights[st])

            #test_rank = self.predict_from_words(self.d.X_test_words[st])

            for pl in range(num_playlists):
                rank_list = test_rank[pl,:].tolist()[0]
                result = get_all_metrics(self.d.test_truth[st][pl], rank_list, self.num_predictions)
                results[st][pl] = np.array(result)

                    #  ignores test set songs not found in training set
            pbar.update(1)
        pbar.close()

        average_result = np.mean(results, axis=1)

        print("Number Training Playlists and Tracks:", self.d.X.shape)
        print("LSA dims: ", self.num_components)
        print("LSA Track Corpus Size:", self.lsa_track_mask.size, "(min track prior =", self.min_track_prior,")")
        print()
        self.print_subtest_results(self.d.subtest_name, metric_names, average_result)
        print()
        self.print_overall_results(metric_names, np.mean(average_result, axis=0))

        return average_result


    def generate_submission(self, filepath, weights, z_score=False):

        print("Encoding and Recoding Challenge Set Matrix")

        f = open(filepath, 'w')
        f.write("team_info,main,JimiLab,dougturnbull@gmail.com\n")

        num_subtest = 10
        num_playlists = len(self.d.challenge)               # 10000
        subtest_size = int(len(self.d.challenge) / num_subtest)  # 1000

        #rank = np.zeros(num_playlists, self.num_predictions)
        # create all popularity vecs so that 1st place is pop of 1.0
        pop_vec = self.d.popularity_vec / np.max(self.d.popularity_vec)

        pbar = tqdm(total=num_subtest)
        pbar.write('~~~~~~~ Generating Ranks by Subchallenge ~~~~~~~')
        for i in range(num_subtest):

            start = i*subtest_size
            end = start+subtest_size
            rank = self.predict_from_matrices(self.d.X_challenge[start:end, :].tocsc(),
                                              pop_vec,
                                              self.d.X_challenge_top_tracks[start:end, :],
                                              self.d.X_challenge_words[start:end, :],
                                              weights[i])

            (num_rows, num_columns) = rank.shape
            for pid in range(num_rows):

                spotify_pid = self.d.pid_to_spotify_pid[start+pid]
                f.write(str(spotify_pid))

                for tid in range(num_columns):
                    track_id = rank[pid, tid]
                    f.write("," + str(self.d.id_to_uri[track_id][0]))
                f.write("\n")

            pbar.update(1)

        pbar.close()
        f.close()


if __name__ == '__main__':

    """ Parameters for Loading Data """
    generate_data_arg = False   # True - load data for given parameter settings
    #                             False - only load data if pickle file doesn't already exist
    create_pickle_file_arg = True     #create a pickle file
    train_size_arg = 100000      # number of playlists for training
    test_size_arg = 2000        # number of playlists for testing
    load_challenge_arg = True   # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge dat (should always be True)
    random_baseline_arg = False  # set to true if you want to run random baseline

    num_components_arg = 128
    lsa_min_track_prior_arg = 0.0002  # minimum prior probability needed to keep track in LSA training matrix size (default 0.0002 or 2 / 10000 playlists
    lsa_zscore_arg = True             # zscore the output of the LSA weight after embedding and projecting back into the original space

    lsa_weight_arg =  .2           # weight of LSA in linear combination
    popularity_weight_arg = .001    # set to 0 for no popularity bias, set to 1 for popularity baseline
    related_track_weight_arg = .6  # weight for top tracks from albums and artists already in the playlist
    words_weight_arg = .2

    weights = (lsa_weight_arg, popularity_weight_arg, related_track_weight_arg, words_weight_arg)

    sub_weights = [(0.9, 0.01, 0.0001, 0.08989999999999998),
                   (0.2, 0.0, 0.001, 0.799),
                   (0.2, 0.0, 0.001, 0.799),
                   (0.1, 0.2, 0.6, 0.09999999999999998),
                   (0.1, 0.2, 0.6, 0.09999999999999998),
                   (0.1, 0.2, 0.6, 0.09999999999999998),
                   (0.1, 0.2, 0.6, 0.09999999999999998),
                   (0.1, 0.2, 0.6, 0.09999999999999998),
                   (0.4, 0.0, 0.6, 0.0),
                   (0.1, 0.2, 0.6, 0.09999999999999998)]

    submission_file_arg = os.path.join(os.getcwd(), 'data/submissions/lsa_test_June20.csv')

    print("Starting Program")
    d = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg,
                  generate_data_arg, create_pickle_file_arg)

    lsa = PredictWithLSA(d, num_components=num_components_arg, lsa_min_track_prior=lsa_min_track_prior_arg)
    lsa.learn_model()


    if False:
        weight_arr = [(.25, .25, .25, .25),
                  (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0,0,0,1),
                  (.7, .1, .1, .1), (.1, .7, .1, .1),(.1, .1, .7, .1),(.1, .1, .1, .7),
                  (.4, .1, .1, .4), (.4, .1, .4, .1), (.4, .4, .1, .1), (.1, .4, .4, .1),
                  (.1, .4, .1, .4), (.1, .1, .1, .4),
                  (.3, .3, .3, .1), (.3, .3, .1, .3), (.3, .1, .3, .3), (.1, .3, .3, .3)]
        for weights in weight_arr:
            lsa.predict_playlists(weights, z_score=lsa_zscore_arg)

    if False:
        filename = "data/pickles/test_results50.pickle"
        num_trials = 50
        weight_options = [0.0, 0.0, 0.0001, 0.001, 0.01, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 0.99, 0.999, 0.9999, 1.0, 1.0]
        weight_list = []
        results = np.zeros((10, 3, num_trials), dtype=float)

        for i in range(num_trials):
            lsa_w = random.choice(weight_options)
            while (True):
                pop_w = random.choice(weight_options)
                if pop_w + lsa_w <= 1.0:
                    break
            while (True):
                rt_w = random.choice(weight_options)
                if pop_w + lsa_w + rt_w  <= 1.0:
                    break

            weights = (max(0.0, lsa_w), max(0.0, pop_w), max(0.0, rt_w), max(0.0, 1-lsa_w-pop_w-rt_w) )
            print("\nTrial: ", i)
            results[:, :, i] = lsa.predict_playlists(weights, z_score=lsa_zscore_arg)
            weight_list.append(weights)

            joblib.dump([weight_list, results], filename)

        ncdg = results[:, 1, :]

        top_score = -1 * np.sort(-1 * ncdg, axis=1)[:, 0]
        top_idx = np.argsort(-1 * ncdg, axis=1)[:, 0]

        for i in range(top_idx.shape[0]):
            print(i, top_idx[i], top_score[i], weights[top_idx[i]])
            # print(weights[top_idx[i]])
        print(np.mean(top_score))

    if True:
        lsa.predict_playlists(sub_weights, z_score=lsa_zscore_arg, random_baseline=random_baseline_arg)


    if load_challenge_arg:
        print("Generating Submission file:", submission_file_arg)
        lsa.generate_submission(submission_file_arg, sub_weights, z_score=lsa_zscore_arg)

    print("done")



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
from bayes_opt import BayesianOptimization


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

        """ weights = (lsa_weight, popularity_weight, related_tracks_weight, word_weight)
        weights can either be a tuple with 4 weights, or a list of 10 tuples of 4 weights each
        """

        #print("\nStarting playlist prediction...")
        print("Weights (LSA, Pop, Related Track, Title Words):", weights )


        num_subtest = len(self.d.test)
        num_playlists = len(self.d.test[0])
        metric_names = ["r_prec", "ndcg", "clicks"]
        num_metrics = len(metric_names)

        results = np.zeros((num_subtest,num_playlists, num_metrics), dtype=float)

        # create all popularity vecs so that 1st place is pop of 1.0
        pop_vec = self.d.popularity_vec / np.max(self.d.popularity_vec)

        #pbar = tqdm(total=num_subtest)
        #pbar.write('~~~~~~~ Predicting Playlists ~~~~~~~')

        for st in range(num_subtest):

            if type(weights) == list:
                w = weights[st]
            else:
                w = weights

            test_rank = self.predict_from_matrices(self.d.X_test[st].tocsc(),
                                                   pop_vec,
                                                   self.d.X_test_top_tracks[st],
                                                   self.d.X_test_words[st],
                                                   w)

            #test_rank = self.predict_from_words(self.d.X_test_words[st])

            for pl in range(num_playlists):
                rank_list = test_rank[pl,:].tolist()[0]
                result = get_all_metrics(self.d.test_truth[st][pl], rank_list, self.num_predictions)
                results[st][pl] = np.array(result)

                    #  ignores test set songs not found in training set
            #pbar.update(1)
        #pbar.close()

        average_result = np.mean(results, axis=1)

        print("Number Training Playlists and Tracks:", self.d.X.shape)
        print("Min Track Prior ", self.d.min_track_prior)
        print("LSA dims: ", self.num_components)
        print("LSA Track Corpus Size:", self.lsa_track_mask.size, "(LSA min track prior =", self.min_track_prior,")")
        print()
        self.print_subtest_results(self.d.subtest_name, metric_names, average_result)
        print()
        self.print_overall_results(metric_names, np.mean(average_result, axis=0))

        return average_result

    def predict_playlists_bayes(self, st, w0, w1, w2, w3):

        st = int(st) # repace this later

        num_playlists = len(self.d.test[0])
        metric_names = ["r_prec", "ndcg", "clicks"]
        num_metrics = len(metric_names)

        results = np.zeros((num_playlists, num_metrics), dtype=float)

        # create all popularity vecs so that 1st place is pop of 1.0
        pop_vec = self.d.popularity_vec / np.max(self.d.popularity_vec)

        w = (w0, w1, w2, w3)


        test_rank = self.predict_from_matrices(self.d.X_test[st].tocsc(),
                                                pop_vec,
                                                self.d.X_test_top_tracks[st],
                                                self.d.X_test_words[st],
                                                w)

        for pl in range(num_playlists):
            rank_list = test_rank[pl, :].tolist()[0]
            result = get_all_metrics(self.d.test_truth[st][pl], rank_list, self.num_predictions)
            results[pl] = np.array(result)

        average_result = np.mean(results, axis=0)

        return average_result[1]





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
            if type(weights) == list:
                w = weights[i]
            else:
                w = weights


            rank = self.predict_from_matrices(self.d.X_challenge[start:end, :].tocsc(),
                                              pop_vec,
                                              self.d.X_challenge_top_tracks[start:end, :],
                                              self.d.X_challenge_words[start:end, :],
                                              w)

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
    generate_data_arg = True   # True - load data for given parameter settings
    #                             False - only load data if pickle file doesn't already exist
    create_pickle_file_arg = True     #create a pickle file
    train_size_arg = 500000      # number of playlists for training
    test_size_arg = 5000        # number of playlists for testing
    load_challenge_arg = True  # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge dat (should always be True)
    random_baseline_arg = False  # set to true if you want to run random baseline
    min_track_prior_arg = 0.0001
    text_index_text_mode_arg = "ntn"

    num_components_arg = 128
    lsa_min_track_prior_arg = 0.0002  # minimum prior probability needed to keep track in LSA training matrix size (default 0.0002 or 2 / 10000 playlists
    lsa_zscore_arg = True             # zscore the output of the LSA weight after embedding and projecting back into the original space

    lsa_weight_arg =  .4          # weight of LSA in linear combination
    popularity_weight_arg = 0.0001    # set to 0 for no popularity bias, set to 1 for popularity baseline
    related_track_weight_arg = .4  # weight for top tracks from albums and artists already in the playlist
    words_weight_arg = .2

    weights = (lsa_weight_arg, popularity_weight_arg, related_track_weight_arg, words_weight_arg)

    a = [(0.4, 0.3, 0.1, 0.2),  #100 per subtest
         (0.3, 0.1, 0.2, 0.4),
         (0.5, 0.0, 0.3, 0.2),
         (0.4, 0.0, 0.4, 0.2),
         (0.4, 0.0, 0.4, 0.2),
         (0.5, 0.0, 0.3, 0.2),
         (0.5, 0.0, 0.3, 0.2),
         (0.4, 0.0, 0.6, 0.0),
         (0.8, 0.0, 0.0, 0.2),
         (0.4, 0.0, 0.6, 0.0)]

    b = [(0.6, 0.15, 0.2, 0.05),  # 400 per subtest
         (0.25, 0.0001, 0.6, 0.15),
         (0.8, 0.0001, 0.01, 0.16),
         (0.25, 0.0001, 0.6, 0.15),
         (0.7, 0.2, 0.01, 0.09),
         (0.25, 0.0001, 0.6, 0.15),
         (0.8, 0.0001, 0.01, 0.19),
         (0.6, 0.15, 0.2, 0.05),
         (0.6, 0.15, 0.2, 0.05),
         (0.95, 0.0, 0.01, 0.04)]

    c = [(1, 1, 0.47, 0.35),        #optimized
         (0.5, 0.2, 0.88, 0.37),
         (1, 1, 1, 0.25),
         (1, 0, 0.61, 0.52),
         (0.14, 0.0025, 0.7121, 0.058),
         (0.63, 0.1, 0.8, 0.15),
         (0.79, 0.41, 0.54, 0.13),
         (0.49, 0, 1, 0),
         (0.63, 0.99, 0.99, 0.04),
         (0.61, 0, 1, 0)]

    dd = [(0.35, 0.35, 0.17, 0.12), #optimized and normalized
         (0.26, 0.10, 0.45, 0.19),
         (0.31, 0.31, 0.31, 0.08),
         (0.47, 0.00, 0.29, 0.24),
         (0.15, 0.00, 0.78, 0.06),
         (0.38, 0.06, 0.48, 0.09),
         (0.42, 0.22, 0.29, 0.07),
         (0.33, 0.00, 0.67, 0.00),
         (0.24, 0.37, 0.37, 0.02),
         (0.38, 0.00, 0.62, 0.00)]

    e = [(0.0, 0.3, 0.1, 0.2),  # doug guess
         (0.3, 0.1, 0.2, 0.4),
         (0.5, 0.0, 0.3, 0.2),
         (0.5, 0.0, 0.5, 0.0),
         (0.4, 0.0, 0.4, 0.2),
         (0.5, 0.0, 0.5, 0.0),
         (0.5, 0.0, 0.3, 0.1),
         (0.5, 0.0, 0.4, 0.1),
         (0.5, 0.0, 0.4, 0.1),
         (0.5, 0.0, 0.4, 0.1)]

    f = [(0.0, 0.5, 0.3, 0.2),  # doug guess simplfied
         (0.3, 0.1, 0.2, 0.4),
         (0.5, 0.0, 0.3, 0.2),
         (0.5, 0.0, 0.5, 0.0),
         (0.4, 0.0, 0.4, 0.2),
         (0.5, 0.0, 0.5, 0.0),
         (0.6, 0.0, 0.4, 0.0),
         (0.6, 0.0, 0.4, 0.0),
         (0.6, 0.0, 0.4, 0.0),
         (0.6, 0.0, 0.4, 0.0)]

    fiveK_weights = [(0.48270270281836813, 0.7448876242714548, 0.8873458428769633, 0.15564998404090447),
     (0.6665980154381933, 0.9053823615161176, 0.4117130073449573, 0.2148710518378656),
     (0.8827692081275599, 0.5576141929834891, 0.49192775259341104, 0.2999736449122169),
     (0.8800370593956184, 0.7937380143368223, 0.8841046630093821, 0.34700353058398903),
     (0.5274603443643752, 0.07455477305947611, 0.1880354271110969, 0.03071420816074444),
     (0.6307804397623651, 0.27749035743731953, 0.7761038220705893, 0.06690470605221444),
     (0.9193785447942945, 0.6314566605491208, 0.716798086280039, 0.13545127867094608),
     (0.5828181810021488, 0.970491938366122, 0.7521723287576919, 0.02099917789974426),
     (0.6775291332800575, 0.5180995363786292, 0.7337840488893119, 0.029505250640784464),
     (0.9999999982720098, 2.158247870415833e-09, 1.0, 0.0)]

    super_weights = [dd]


    submission_file_arg = os.path.join(os.getcwd(), 'data/submissions/lsa_test_June28_600K.csv')

    print("Starting Program")
    d = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg,
                  generate_data_arg, create_pickle_file_arg, text_index_text_mode_arg, min_track_prior_arg)

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
        num_trials = 25
        weight_options = [.0, .0001, .001, .01, .05, 0.1, 0.15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95, .99, .999, .9999, 1.0]
        weight_list = []
        results = np.zeros((10, 3, num_trials), dtype=float)
        best_weights = list()

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
            print(i, top_idx[i], top_score[i], weight_list[top_idx[i]])
            # print(weight_list[top_idx[i]])
            best_weights.append(weight_list[top_idx[i]])
        print(np.mean(top_score))
        print("Results with Best Weights:")
        lsa.predict_playlists(best_weights, z_score=lsa_zscore_arg, random_baseline=random_baseline_arg)
        print(best_weights)

    if True:  #Bayesian Optimation:
        best_weights = list()
        #ncdg = lsa.predict_playlists_bayes(0, .33, .33, .33)
        for st in range(10):

            bo = BayesianOptimization(lsa.predict_playlists_bayes,  {'st':(st,st),
                                                                     'w0': (0,1), 'w1':(0,1),
                                                                     'w2':(0,1), 'w3':(0,1)})
            bo.maximize(init_points=20, n_iter=5, acq='ucb', kappa=5)
            print(bo.res['max'])
            d = bo.res['max']
            p = d['max_params']
            best_weights.append((p['w0'], p['w1'], p['w2'], p['w3']))

        lsa.predict_playlists(best_weights, z_score=lsa_zscore_arg, random_baseline=random_baseline_arg)
        print(best_weights)

    if False:
        for sub_weights in super_weights:
            print(sub_weights)
            lsa.predict_playlists(sub_weights, z_score=lsa_zscore_arg, random_baseline=random_baseline_arg)


    if load_challenge_arg:

        #best_weights = [(0.48270270281836813, 0.7448876242714548, 0.8873458428769633, 0.15564998404090447), (0.6665980154381933, 0.9053823615161176, 0.4117130073449573, 0.2148710518378656), (0.8827692081275599, 0.5576141929834891, 0.49192775259341104, 0.2999736449122169), (0.8800370593956184, 0.7937380143368223, 0.8841046630093821, 0.34700353058398903), (0.5274603443643752, 0.07455477305947611, 0.1880354271110969, 0.03071420816074444), (0.6307804397623651, 0.27749035743731953, 0.7761038220705893, 0.06690470605221444), (0.9193785447942945, 0.6314566605491208, 0.716798086280039, 0.13545127867094608), (0.5828181810021488, 0.970491938366122, 0.7521723287576919, 0.02099917789974426), (0.6775291332800575, 0.5180995363786292, 0.7337840488893119, 0.029505250640784464), (0.9999999982720098, 2.158247870415833e-09, 1.0, 0.0)]
        #lsa.predict_playlists(best_weights, z_score=lsa_zscore_arg, random_baseline=random_baseline_arg)

        print("Generating Submission file:", submission_file_arg)
        lsa.generate_submission(submission_file_arg, best_weights, z_score=lsa_zscore_arg)

    print("done")


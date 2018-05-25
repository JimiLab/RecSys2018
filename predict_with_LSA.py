
from sklearn.decomposition import TruncatedSVD
from predict import Predict
from DataManager import load_data
import math
import numpy as np
from tqdm import tqdm
import os
from metrics import get_all_metrics


class PredictWithLSA(Predict):

    def __init__(self, data, num_components=64, missing_track_rate=0.2):
        # Call init on super class
        Predict.__init__(self)
        self.d = data  # DataManager Object
        self.num_components = num_components
        self.missing_track_rate = missing_track_rate
        self.num_predictions = 500
        self.svd = TruncatedSVD(n_components=self.num_components)

    def learn_model(self):

        if not hasattr(self.d, 'X'):
            print("Pickle File does not have pre-computed numpy X matrix. Aborting")
            return

        self.svd.fit(self.d.X)

    def predict_playlists(self, popularity_weight=0.0, random_baseline=False):
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
            embedded_test_vecs = self.svd.transform(self.d.X_test[st] )
            test_vecs_hat = self.svd.inverse_transform(embedded_test_vecs)

            # make sure non-missing tracks are not included in the ranking
            test_vecs_hat -= self.d.X_test[st]* -99999999

            test_vecs_hat = (1 - popularity_weight) * test_vecs_hat + popularity_weight * self.d.popularity_vec

            test_rank = np.argsort(-1 * test_vecs_hat, axis=1)

            if random_baseline:  # Change to True for Random Baseline
                np.random.shuffle(test_rank.T)



            test_rank = test_rank[:, 0:self.num_predictions]


            for pl in range(num_playlists):
                rank_list = test_rank[pl,:].tolist()[0]
                result = get_all_metrics(self.d.test_truth[st][pl], rank_list, self.num_predictions )
                results[st][pl] = np.array(result)

                    #  ignores test set songs not found in training set
            pbar.update(1)
        pbar.close()

        average_result = np.mean(results, axis=1)
        print()
        # print("Average Recall @ ", num_predictions,":", np.average(recall_500))
        self.print_subtest_results(self.d.subtest_name, metric_names, average_result)
        print()
        self.print_overall_results(metric_names, np.mean(average_result, axis=0))

    def generate_submission(self, filepath, popularity_weight=0.0):

        print("Encoding and Recoding Challenge Set Matrix")

        f = open(filepath, 'w')
        f.write("team_info,main,JimiLab,dougturnbull@gmail.com\n")

        x_challenge_embedded = self.svd.transform(self.d.x_challenge)
        x_challenge_hat = self.svd.inverse_transform(x_challenge_embedded)

        for i in range(len(self.d.challenge)):
            x_challenge_hat[i, self.d.challenge[i]] = -99999999999

        x_challenge_hat = (1 - popularity_weight) * x_challenge_hat + popularity_weight * self.d.popularity_vec

        rank = np.argsort(-1 * x_challenge_hat, axis=1)
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
    train_size_arg = 10000      # number of playlists for training
    test_size_arg = 1000        # number of playlists for testing
    load_challenge_arg = False   # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge data
    num_components_arg = 32
    popularity_weight_arg = 1   # set to 0 for no popularity bias, set to 1 for popularity baseline
    random_baseline_arg = False  # set to true if you want to run random baseline

    submission_file_arg = os.path.join(os.getcwd(), 'data/submissions/lsa_test.csv')

    d = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg, generate_data_arg)

    lsa = PredictWithLSA(d, num_components=num_components_arg, missing_track_rate=0.2)
    lsa.learn_model()
    lsa.predict_playlists(popularity_weight=popularity_weight_arg, random_baseline=random_baseline_arg)
    if load_challenge_arg:
        print("Generating Submission file:", submission_file_arg)
        lsa.generate_submission(submission_file_arg, popularity_weight=popularity_weight_arg)
    print("done")

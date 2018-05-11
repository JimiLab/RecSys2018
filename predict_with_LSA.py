
from sklearn.decomposition import TruncatedSVD
from predict import predict
from DataManager import load_data
import math
import numpy as np
from tqdm import tqdm
import os


class PredictWithLSA(predict):

    def __init__(self, data, num_components=64, missing_track_rate=0.2):
        # Call init on super class
        predict.__init__(self)
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

    def predict_playlists(self, popularity_weight=0.0):
        print("\nStarting playlist prediction...")

        recall_500 = list()
        r_prec = list()
        ndcg = list()
        song_click = list()

        num_playlists = self.d.test_size

        pbar = tqdm(total=num_playlists)
        pbar.write('~~~~~~~ Predicting Playlists ~~~~~~~')

        for i in range(num_playlists):
            test_vec = self.d.X_test[i, :]
            test_len = len(self.d.test[i])
            nz_idx = test_vec.nonzero()[1]
            num_missing = math.ceil(len(nz_idx) * self.missing_track_rate)
            np.random.shuffle(nz_idx)
            missing_tracks = nz_idx[0:num_missing]
            non_missing_tracks = nz_idx[num_missing:]

            # remove missing tracks before embedding
            test_vec[0, missing_tracks] = 0

            embedded_test_vec = self.svd.transform(test_vec)
            test_vec_hat = self.svd.inverse_transform(embedded_test_vec)

            # make sure non-missing tracks are not included in the ranking
            test_vec_hat[0, non_missing_tracks] = -99999999

            test_vec_hat = (1 - popularity_weight) * test_vec_hat + popularity_weight * self.d.popularity_vec

            test_rank = np.argsort(-1 * test_vec_hat, axis=1)[0, 0:self.num_predictions]

            if len(missing_tracks) > 0:

                extend_amt = int(math.ceil(test_len * self.missing_track_rate) - num_missing)
                gt = list(missing_tracks)
                gt.extend([-1] * extend_amt)

                gt_vec = [0] * self.num_predictions

                test_rank_list = list(test_rank)
                for v in missing_tracks:
                    if v in test_rank_list:
                        gt_vec[test_rank_list.index(v)] = 1
                # Pick up from here
                ndcg_val = self.ncdg(gt_vec, self.num_predictions)  # ndcg_at_k(gt_vec, len(test_rank_list), 0)
                ndcg.append(ndcg_val)
                song_click.append(self.song_clicks_metric(gt_vec))
                r_prec.append(self.r_precision(gt, test_rank))
                recall_500.append(self.precision_and_recall_at_k(gt, test_rank)[1])

                #  ignores test set songs not found in training set
            pbar.update(1)
        pbar.close()
        print()
        # print("Average Recall @ ", num_predictions,":", np.average(recall_500))
        print("Average R Prec:\t", np.round(np.average(r_prec), decimals=3))
        print("Average NDGC:\t", np.round(np.average(ndcg), decimals=3))
        print("Average Clicks\t", np.round(np.average(song_click), decimals=3))
        print("Number Trials:\t", len(recall_500))

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
    train_size_arg = 24000      # number of playlists for training
    test_size_arg = 2000        # number of playlists for testing
    load_challenge_arg = True   # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge data
    num_components_arg = 256
    popularity_weight_arg = 0.0001

    submission_file_arg = os.path.join(os.getcwd(), 'data/submissions/lsa_24KTrain_256Comp.csv')

    d = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg, generate_data_arg)

    lsa = PredictWithLSA(d, num_components=num_components_arg, missing_track_rate=0.2)
    lsa.learn_model()
    lsa.predict_playlists(popularity_weight=popularity_weight_arg)
    print("Generating Submission file:", submission_file_arg)
    lsa.generate_submission(submission_file_arg, popularity_weight=popularity_weight_arg)
    print("done")

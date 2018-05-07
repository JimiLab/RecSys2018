
from sklearn.decomposition import TruncatedSVD
from predict import predict
from DataManager import load_data, DataManager
from rank_metrics import ndcg_at_k
import math
import numpy as np
from tqdm import tqdm

class predict_with_LSA(predict):

    def __init__(self, data, num_components=64, missing_track_rate=0.2):
        self.d = data  #DataManager Object
        self.num_components = num_components
        self.missing_track_rate = missing_track_rate


    def predict_playlists(self):
        print("\nStarting playlist prediction...")

        num_predictions = 500
        svd = TruncatedSVD(n_components=self.num_components)

        if not hasattr(self.d, 'X'):
            print("Pickle File does not have pre-computed numpy X matrix. Aborting")
            return

        svd.fit(self.d.X)

        recall_500 = list()
        r_prec = list()
        ndcg = list()
        song_click = list()

        num_playlists = self.d.test_size

        pbar2 = tqdm(total=num_playlists)
        pbar2.write('~~~~~~~ Predicting Playlists ~~~~~~~')

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

            embedded_test_vec = svd.transform(test_vec)
            test_vec_hat = svd.inverse_transform(embedded_test_vec)

            # make sure non-missing tracks are not included in the ranking
            test_vec_hat[0, non_missing_tracks] = -999999

            test_rank = np.argsort(-1 * test_vec_hat, axis=1)[0, 0:num_predictions]

            if len(missing_tracks) > 0:

                extend_amt = math.ceil(test_len * self.missing_track_rate) - num_missing
                gt = list(missing_tracks)
                gt.extend([-1] * extend_amt)

                gt_vec = [0] * num_predictions

                test_rank_list = list(test_rank)
                for v in missing_tracks:
                    if v in test_rank_list:
                        gt_vec[test_rank_list.index(v)] = 1
                # Pick up from here
                ndcg_val = self.ncdg(gt_vec, num_predictions) #ndcg_at_k(gt_vec, len(test_rank_list), 0)
                ndcg.append(ndcg_val)
                song_click.append(self.song_clicks_metric(gt_vec))
                r_prec.append(self.r_precision(gt, test_rank))
                recall_500.append(self.precision_and_recall_at_k(gt, test_rank)[1])

                #  ignores test set songs not found in training set
            pbar2.update(1)

        print()
        # print("Average Recall @ ", num_predictions,":", np.average(recall_500))
        print("Average R Prec:\t", np.round(np.average(r_prec), decimals=3))
        print("Average NDGC:\t", np.round(np.average(ndcg), decimals=3))
        print("Average Clicks\t", np.round(np.average(song_click), decimals=3))
        print("Number Trials:\t", len(recall_500))



if __name__ == '__main__':

    """ Parameters for Loading Data """
    generate_data = False   # True - load data for given parameter settings
                            # False - only load data if pickle file doesn't already exist
    train_size = 2000      # number of playlists for training
    test_size = 500        # number of playlists for testing
    load_challenge = False  # loads challenge data when creating a submission to contest
    create_matrices = True  # creates numpy matrices for train, test, and (possibly) challenge data

    d = load_data(train_size, test_size, load_challenge, create_matrices, generate_data)

    lsa = predict_with_LSA(d, num_components=64, missing_track_rate=0.2)
    lsa.predict_playlists()
import os
import math
import json
from tqdm import tqdm
import random
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn import metrics
from rank_metrics import ndcg_at_k


class DataManager:
    """
     Loads data from provide playlist JSON Files
     Main data structure is a list of lists: playlist ID, track IDs
       where track ID is found in the dictionary uri_to_id and id_to_uri
    """

    def __init__(self, path, max_number_playlists=10000, train_test_split=0.8):
        self.DATA_DIR = path
        self.data_cache = dict()
        self.max_number_playlists = max_number_playlists
        self.train_test_split = train_test_split
        self.train = defaultdict(list)
        self.test = defaultdict(list)  # tracks not in training set will have id = -1
        self.uri_to_id = dict()
        self.id_to_uri = dict()


    def load_playlist_data(self):
        """
        Loads MPD JSON data files sequentially.

        Create train and test list of lists where each track is
        represented by internal id

        if track does not appear in training set, it is represented with
        an id = -1 in the test set playlist list

        Args:
            None
        Returns:
            None
        """

        num_files_to_load = math.ceil(self.max_number_playlists / 1000)+1
        train_pid = 0
        test_pid = 0
        test_playlist = defaultdict(list)
        tid = 0

        prefix_len = len("spotify:track:")

        pbar = tqdm(total=self.max_number_playlists)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')

        for file in os.listdir(self.DATA_DIR)[:num_files_to_load]:
            if not file.startswith("mpd.slice"):
                continue
            data = json.load(open(self.DATA_DIR + file))
            for playlist in data['playlists']:

                is_train =  random.uniform(0, 1) < self.train_test_split

                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    track_uri = track_uri[prefix_len:]

                    if is_train:
                        # new track that has never been encountered before
                        if track_uri not in self.uri_to_id.keys():
                            self.uri_to_id[track_uri] = tid
                            self.id_to_uri[tid] = (track['track_uri'], track['track_name'],
                                                   track['artist_uri'], track['artist_name'])
                            tid += 1

                        track_id = self.uri_to_id[track_uri]
                        self.train[train_pid].append(track_id)


                    else: #test playlist
                        test_playlist[test_pid].append(track_uri)



                if is_train:
                    train_pid += 1
                else:
                    test_pid += 1
                pbar.update(1)
                if train_pid + test_pid > self.max_number_playlists:
                    break

        pbar.close()

        # resolve test playlist against training track corpus
        #  set unknown tracks to have id = -1
        for test_pid, tracks in test_playlist.items():
            for uri in tracks:
                if uri not in self.uri_to_id.keys():
                    self.test[test_pid].append(-1)
                else:
                    self.test[test_pid].append(self.uri_to_id[uri])



        return


    def pickle_data(self, filename):
        print("Dumping file: ", filename)
        #joblib.dump((self.test, self.train, self.uri_to_id, self.id_to_uri), filename)
        joblib.dump(self, filename)

    def create_matrix(self):

        num_rows = len(self.train)
        num_cols = len(self.id_to_uri)
        self.X = lil_matrix((num_rows, num_cols), dtype=np.int8)

        for pid, playlist in self.train.items():
            for tid in playlist:
                self.X[pid, tid] = 1

        self.X_test = lil_matrix((len(self.test), num_cols), dtype=np.int8)
        for pid, playlist in self.test.items():
            for tid in playlist:
                if tid != -1:
                    self.X_test[pid, tid] = 1

    def precision_and_recall_at_k(self, ground_truth, prediction, k=-1):
        """

        :param ground_truth:
        :param prediction:
        :param k: how far down the ranked list we look, set to -1 (default) for all of the predictions
        :return:
        """

        if (k == -1):
            k = len(prediction)
        prediction = prediction[0:k]

        numer =  len(set(ground_truth).intersection(set(prediction)))
        prec = numer / k
        recall = numer / len(ground_truth)
        return prec, recall


    def r_precision(self, ground_truth, prediction):
        k = len(ground_truth)
        p, r = self.precision_and_recall_at_k(ground_truth, prediction, k)
        return p

    def song_clicks_metric(self, ranking):
        """
        Spotify p
        :param ranking:
        :return:
        """

        if 1 in ranking:
            first_idx = ranking.index(1)

            return math.floor(first_idx/10)
        return 51

    def predict_playlists(self,svd_components=64, missing_track_rate = .2):
        print("\nStarting playlist prediction...")

        num_predictions = 500
        svd = TruncatedSVD(n_components=svd_components)
        svd.fit(self.X)

        recall_500 = list()
        r_prec = list()
        ndcg = list()
        song_click = list()

        num_playlists = min(self.X_test.shape[0],1000)

        for i in range(num_playlists):
            if i % 100 == 0:
                print(i," ", end="")
            test_vec = self.X_test[i, :]
            test_len = len(self.test[i])
            nz_idx = test_vec.nonzero()[1]
            num_missing = math.ceil(len(nz_idx)*(missing_track_rate))
            np.random.shuffle(nz_idx)
            missing_tracks = nz_idx[0:num_missing]
            non_missing_tracks = nz_idx[num_missing:]

            # remove missing tracks before embedding
            test_vec[0,missing_tracks] = 0

            embedded_test_vec = svd.transform(test_vec)
            test_vec_hat = svd.inverse_transform(embedded_test_vec)

            # make sure non-missing tracks are not included in the ranking
            test_vec_hat[0,non_missing_tracks] = -999999

            test_rank = np.argsort(-1*test_vec_hat, axis=1)[0,0:num_predictions]

            if len(missing_tracks) > 0:

                extend_amt = math.ceil(test_len*missing_track_rate) - num_missing
                gt = list(missing_tracks)
                gt.extend([-1]*extend_amt)

                gt_vec = [0]*num_predictions

                test_rank_list = list(test_rank)
                for v in missing_tracks:
                    if v in test_rank_list:
                        gt_vec[test_rank_list.index(v)] = 1
                # Pick up from here
                ndcg_val = ndcg_at_k(gt_vec,len(test_rank_list), 0)
                ndcg.append(ndcg_val)
                song_click.append(self.song_clicks_metric(gt_vec))
                r_prec.append(self.r_precision(gt, test_rank))
                recall_500.append(self.precision_and_recall_at_k(gt, test_rank)[1])


                                                                                  #  ignores test set songs not found in training set
        print()
        #print("Average Recall @ ", num_predictions,":", np.average(recall_500))
        print("Average R Prec:\t", np.round(np.average(r_prec), decimals=3))
        print("Average NDGC:\t",   np.round(np.average(ndcg),decimals=3))
        print("Average Clicks\t",  np.round(np.average(song_click),decimals=3))
        print("Number Trials:\t", len(recall_500))








if __name__ == '__main__':

    path = os.path.join(os.getcwd(), 'data/mpd.v1/data/')
    p_file = os.path.join(os.getcwd(),'data/pickles/MPD_20K.pkl')
    generate_data = False


    if generate_data:
        d = DataManager(path, max_number_playlists=40000, train_test_split=0.9)
        d.load_playlist_data()
        d.create_matrix()
        d.pickle_data(p_file)
    else:
        d = joblib.load(p_file)

    print("Train Set Size:", len(d.train))
    for nc in [128]:
       print("\nNumber of SVD Components", nc,"\t", end="")
       d.predict_playlists(svd_components=nc)
    #d.predict_playlists(svd_components=64)

    pass
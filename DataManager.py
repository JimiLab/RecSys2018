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


    def predict_playlists(self,svd_components=64, missing_track_rate = .2):

        num_predictions = 500
        svd = TruncatedSVD(n_components=svd_components)
        svd.fit(self.X)

        recall_500 = list()

        for i in range(self.X_test.shape[0]):
            test_vec = self.X_test[i, :]
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

            cnt = 0
            for j in range(num_predictions):
                v = test_rank[j]
                if v in missing_tracks:
                    #print("Found a good one:", i, v)
                    cnt += 1
                if v in non_missing_tracks:
                    print("Duplicate in non missing track value",i,v)
            #print("Number of good ones:", cnt, " out of ", len(missing_tracks))
            if len(missing_tracks) > 0:
                recall_500.append((cnt)/len(missing_tracks))

        print("Average Recall @ 500:", np.average(recall_500))








if __name__ == '__main__':

    path = os.path.join(os.getcwd(), 'data/mpd.v1/data/')
    p_file = os.path.join(os.getcwd(),'data/pickles/MPD_1K.pkl')
    generate_data = False


    if generate_data:
        d = DataManager(path, max_number_playlists=1000)
        d.load_playlist_data()
        d.create_matrix()
        d.pickle_data(p_file)
    else:
        d = joblib.load(p_file)

    #for nc in [2,4,8,16,32,64,128,256,512, 1024]:
    #   print("Number of SVD Components", nc,"\t", end="")
    #   d.predict_playlists(svd_components=nc)
    d.predict_playlists(svd_components=64)

    pass
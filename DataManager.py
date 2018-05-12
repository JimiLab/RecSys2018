import os
import math
import json
from tqdm import tqdm
import random
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib
from scipy.sparse import lil_matrix


class DataManager:
    """
     Sets up Empty Data Manager object
     Main data structure is three list of lists: playlist ID, track IDs
       where track ID is found in the dictionary uri_to_id and id_to_uri
     Each list represents the three parts of the data set: train, test, challenge
    """

    def __init__(self, path, train_size=10000, test_size=2000, challenge_file=None):
        self.DATA_DIR = path
        self.CHALLENGE_FILE = challenge_file
        self.data_cache = dict()
        self.train_size = train_size
        self.train = defaultdict(list)
        self.test_size = test_size
        self.test = defaultdict(list)           # tracks not in training set will have id = -1
        self.challenge = defaultdict(list)
        self.challenge_size = -1
        self.uri_to_id = dict()
        self.id_to_uri = dict()
        self.track_count = list()
        self.pid_to_spotify_pid = dict()
        self.X = None
        self.X_test = None
        self.popularity_vec = None
        self.X_challenge = None

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
        total_size = self.train_size+self.test_size
        train_test_ratio = self.test_size / total_size

        num_files_to_load = 1000
        # num_files_to_load = math.ceil(total_size / 1000)+1
        train_pid = 0
        test_pid = 0
        test_playlist = defaultdict(list)
        tid = 0

        prefix_len = len("spotify:track:")

        pbar = tqdm(total=self.train_size)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')

        for file in os.listdir(self.DATA_DIR)[:num_files_to_load]:
            if train_pid >= self.train_size and test_pid >= self.test_size:
                break
            if not file.startswith("mpd.slice"):
                continue
            data = json.load(open(self.DATA_DIR + file))
            for playlist in data['playlists']:

                # break if we have enough data

                is_train = random.uniform(0, 1) < train_test_ratio

                if train_pid >= self.train_size and test_pid >= self.test_size:
                    break

                # skip playlist if we have already loaded enough of them for either train or test
                if is_train and train_pid >= self.train_size:
                    continue
                if not is_train and test_pid >= self.test_size:
                    continue

                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    track_uri = track_uri[prefix_len:]

                    if is_train:
                        # new track that has never been encountered before
                        if track_uri not in self.uri_to_id.keys():
                            self.uri_to_id[track_uri] = tid
                            self.id_to_uri[tid] = (track['track_uri'], track['track_name'],
                                                   track['artist_uri'], track['artist_name'])
                            self.track_count.append(0)
                            tid += 1

                        track_id = self.uri_to_id[track_uri]
                        self.train[train_pid].append(track_id)
                        self.track_count[track_id] += 1

                    else:  # test playlist - don't add track_uri if it is only ever encountered with from test tracks
                        test_playlist[test_pid].append(track_uri)

                if is_train:
                    train_pid += 1
                else:
                    test_pid += 1

                pbar.update(1)

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

    def load_challenge_data(self):
        data = json.load(open(self.CHALLENGE_FILE))
        prefix_len = len("spotify:track:")

        cid = 0

        pbar = tqdm(total=10000)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')

        for playlist in data['playlists']:
            self.pid_to_spotify_pid[cid] = playlist['pid']
            self.challenge[cid] = list()
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                track_uri = track_uri[prefix_len:]

                if track_uri not in self.uri_to_id.keys():
                    self.challenge[cid].append(-1)
                else:
                    self.challenge[cid].append(self.uri_to_id[track_uri])

            cid += 1
            pbar.update(1)
        self.challenge_size = len(self.challenge)
        pbar.close()

    def pickle_data(self, filename):
        # Use file handle to ensure file exists upon serialization
        with open(filename, 'wb') as file:
            joblib.dump(self, file)

    def create_train_matrix(self):

        num_rows = len(self.train)
        num_cols = len(self.id_to_uri)

        self.X = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for pid, playlist in self.train.items():
            for tid in playlist:
                self.X[pid, tid] = 1

    def create_test_matrix(self):
        num_rows = len(self.test)
        num_cols = len(self.id_to_uri)
        self.X_test = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for pid, playlist in self.test.items():
            for tid in playlist:
                if tid != -1:
                    self.X_test[pid, tid] = 1

    def create_challenge_matrix(self):
        num_rows = len(self.challenge)
        num_cols = len(self.id_to_uri)
        self.X_challenge = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for pid, playlist in self.challenge.items():
            for tid in playlist:
                if tid != -1:
                    self.X_challenge[pid, tid] = 1

    def calculate_popularity(self):
        self.popularity_vec = np.array(self.track_count) / self.train_size

    def create_matrices(self):
        self.create_train_matrix()
        self.create_test_matrix()
        if self.CHALLENGE_FILE is not None:
            self.create_challenge_matrix()

# END OF CLASS


def load_data(train_size=10000, test_size=2000, load_challenge=False, create_matrices=False, generate_data=False):

    """ Fixed Path Names """
    data_folder = os.path.join(os.getcwd(), 'data/mpd.v1/data/')
    challenge_file = os.path.join(os.getcwd(), 'data/challenge.v1/challenge_set.json')
    pickle_folder = os.path.join(os.getcwd(), 'data/pickles/')

    c_str = ""
    c_file = None
    if load_challenge:
        c_str = "_with_challenge"
        c_file = challenge_file

    m_str = ""
    if create_matrices:
        m_str = "_with_matrices"

    pickle_file = pickle_folder + "MPD_" + str(math.floor(train_size/1000.0)) + "KTrain_" + \
                  str(math.floor(test_size / 1000.0)) + \
                  "KTest" + c_str + m_str + ".pickle"

    pickle_exists = os.path.isfile(pickle_file)

    if generate_data or not pickle_exists:
        d = DataManager(data_folder, train_size=train_size, test_size=test_size, challenge_file=c_file)
        print("Load Playlist Data")
        d.load_playlist_data()
        if load_challenge:
            print("Load Challenge Set Data")
            d.load_challenge_data()
        if create_matrices:
            print("Calculate Numpy Matrices")
            d.create_matrices()
        d.calculate_popularity()
        print("Pickle Data into file: "+pickle_file)
        d.pickle_data(pickle_file)
    else:
        print("Load data from Pickle File: "+pickle_file)
        d = joblib.load(pickle_file)
    return d


if __name__ == '__main__':

    """ Parameters for Loading Data """
    generate_data_arg = True    # True - load data for given parameter settings
    #                             False - only load data if pickle file doesn't already exist
    train_size_arg = 2000       # number of playlists for training
    test_size_arg = 1000        # number of playlists for testing
    load_challenge_arg = False  # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge data

    data_in = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg, generate_data_arg)

    # lsa = predict_with_LSA()
    # lsa.predict_playlists(svd_components=64)

    pass

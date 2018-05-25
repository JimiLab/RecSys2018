import os
import math
import json
from tqdm import tqdm
import random
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib
from scipy.sparse import lil_matrix

import re


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
        self.train = []
        self.train_title = []
        self.train_description = []

        self.test_size = math.ceil(test_size/10.0)*10 # needs to be a multiple of 10
        self.subtest_size = self.test_size /10
        self.test = []              # all test lists are of size 10 for each of the 10 subchallanges
        self.test_uri = []
        self.test_truth = []
        self.test_truth_uri = []
        self.test_title = []

        for i in range(10):
            self.test.append([])
            self.test_uri.append([])
            self.test_truth.append([])
            self.test_truth_uri.append([])
            self.test_title.append([])



        self.subtest_name = ["title only", "title / first",
                               "title / first 5", "first 5",
                               "title / first 10", "first 10 ",
                               "title / first 25", "title / random 25 ",
                               "title / first 100", "title / random 100" ]
        self.subtest_setup = [(True, 0, True),  (True, 1, True),
                                (True, 5, True),  (False, 5, True),
                                (True, 10, True), (False, 10, True),
                                (True, 25, True), (True, 25, False),
                                (True, 100, True), (True, 100, False)] # (has_title, num_tracks, first_or_random)

        self.challenge = []
        self.challenge_title = []
        self.uri_to_id = dict()
        self.id_to_uri = dict()
        self.track_frequency = []
        self.artist_to_track_id = defaultdict(list)
        self.album_to_track_id = defaultdict(list)
        self.pid_to_spotify_pid = []
        self.X = None
        self.X_test = None
        self.X_challenge = None
        self.popularity_vec = None
        self.prefix = "spotify:track:"

    def normalize_name(self, name):
        name = name.lower()
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _add_train_playlist(self, playlist):

        pid = len(self.train)
        self.train.append([])
        self.train_title.append(self.normalize_name(playlist["name"]))
        description = ""
        if "description" in playlist:
            description = self.normalize_name(playlist["description"])
        self.train_description.append(description)

        modified = playlist["modified_at"]

        for track in playlist['tracks']:
            track_uri = track['track_uri']
            track_uri = track_uri[len(self.prefix):]

            # new track that has never been encountered before
            if track_uri not in self.uri_to_id.keys():
                tid = len(self.id_to_uri)


                self.uri_to_id[track_uri] = tid
                self.id_to_uri[tid] = [track['track_uri'], track['track_name'],
                                       track['artist_uri'], track['artist_name'],
                                       track['album_uri'], track['album_name'],
                                       modified] # will keep time when first appears in a playlist
                self.track_frequency.append(0)



                self.artist_to_track_id[track['artist_uri']].append(tid)
                self.album_to_track_id[track['album_uri']].append(tid)

            track_id = self.uri_to_id[track_uri]
            self.train[pid].append(track_id)
            self.track_frequency[track_id] += 1
            if modified < self.id_to_uri[track_id][6]:  # update time stamp for track if it appears earlier
                self.id_to_uri[track_id][6] = modified






    def _add_test_playlist(self, playlist):

        subtest = random.randint(0,9)

        # if subtest is already full
        if len(self.test_uri[subtest]) >= self.subtest_size:
            return

        num_tracks = playlist["num_tracks"]

        # not enough tracks to hid any tracks
        if num_tracks <= self.subtest_setup[subtest][1]:
            return

        pid = len(self.test[subtest])
        self.test_title[subtest].append(self.normalize_name(playlist["name"]))

        uri_list = list()
        for track in playlist['tracks']:
            track_uri = track['track_uri']
            track_uri = track_uri[len(self.prefix):]
            uri_list.append(track_uri)

        #random tracks from playlist
        if self.subtest_setup[subtest][2] == False:
            random.shuffle(uri_list)

        # number of tracks in the playlist
        split = self.subtest_setup[subtest][1]

        self.test_uri[subtest].append(uri_list[0:split])
        self.test_truth_uri[subtest].append(uri_list[split:])
        pass


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

        train_done = False
        test_done = False

        pbar = tqdm(total=self.train_size)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')

        for file in os.listdir(self.DATA_DIR)[:num_files_to_load]:

            if train_done and test_done:
                break

            if not file.startswith("mpd.slice"):
                continue

            data = json.load(open(self.DATA_DIR + file))

            for playlist in data['playlists']:

                # break if we have enough data
                if train_done and test_done:
                    break

                is_train = random.uniform(0, 1) < train_test_ratio

                # skip playlist if we have already loaded enough of them for either train or test
                if is_train and train_done:
                    continue
                if not is_train and test_done:
                    continue

                if is_train:
                    self._add_train_playlist(playlist)
                    train_done = len(self.train) >= self.train_size
                    pbar.update(1)

                else:
                    self._add_test_playlist(playlist)
                    test_done = True
                    for i in range(10):
                        if len(self.test_uri[i]) < self.subtest_size:
                            test_done = False
                            break



        pbar.close()


        # resolve test playlist against training track corpus
        #  set unknown tracks to have id < 0 (e.g.,  -1, -2, -3, ...
        for s in range(10):
            miss_idx = -1
            for p in range(len(self.test_uri[s])):
                self.test[s].append([])
                self.test_truth[s].append([])
                for uri in self.test_uri[s][p]:
                   if uri not in self.uri_to_id.keys():
                       self.test[s][p].append(-1)
                   else:
                       self.test[s][p].append(self.uri_to_id[uri])

                for uri in self.test_truth_uri[s][p]:
                   if uri not in self.uri_to_id.keys():
                       self.test_truth[s][p].append(miss_idx)
                       miss_idx -= 1
                   else:
                       self.test_truth[s][p].append(self.uri_to_id[uri])
        return

    def load_challenge_data(self):
        data = json.load(open(self.CHALLENGE_FILE))

        pbar = tqdm(total=10000)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')

        for playlist in data['playlists']:
            self.pid_to_spotify_pid.append(playlist['pid'])
            if 'name' in playlist:
                self.challenge_title.append(self.normalize_name(playlist['name']))
            else:
                self.challenge_title.append("")
            track_ids = list()
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                track_uri = track_uri[len(self.prefix):]

                if track_uri not in self.uri_to_id.keys():
                    track_ids.append(-1)
                else:
                    track_ids.append(self.uri_to_id[track_uri])

            self.challenge.append(track_ids)

            pbar.update(1)
        self.challenge_size = len(self.challenge)
        pbar.close()

    def pickle_data(self, filename):
        joblib.dump(self, filename)


    ####  NEED TO UPDATE LSA MATRICES  to use lsa_track_mask -> tracks with not small prior probabilities
    def create_train_matrix(self):

        num_rows = len(self.train)
        num_cols = len(self.lsa_track_mask)

        self.X = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for p in range(num_rows):
            for t in self.train[p]:
                if t in self.lsa_track_mask:
                    self.X[p, self.lsa_id_to_column[t]] = 1

    def create_test_matrix(self):
        num_subtest = len(self.test)
        num_rows = len(self.test[0])
        num_cols = len(self.lsa_track_mask)
        self.X_test = list()
        for s in range(num_subtest):
            mat = lil_matrix((num_rows, num_cols), dtype=np.int8)
            for p in range(num_rows):
                for t in self.test[s][p]:
                    if t >= 0 and t in self.lsa_track_mask:
                        mat[p, self.lsa_id_to_column[t]] = 1
            self.X_test.append(mat)
        return

    def create_challenge_matrix(self):
        num_rows = len(self.challenge)
        num_cols = len(self.lsa_track_mask)
        self.X_challenge = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for p in range(num_rows):
            for t in self.challenge[p]:
                if t >= 0 and t in self.lsa_track_mask:
                    self.X_challenge[p, self.lsa_id_to_column[t]] = 1

    def calculate_popularity(self, top_k = 3):
        print("Calculating Track Prior Proabability, Top Artist Tracks, and Top Album Tracks ")
        self.popularity_vec = np.array(self.track_frequency) / self.train_size
        self.artist_top_tracks = defaultdict(list)
        for k,v in self.artist_to_track_id.items():
            track_pops = self.popularity_vec[v]
            idx = np.argsort(1 / track_pops)[0:min(top_k, len(track_pops))].tolist() #sort artist track by popularity
            for i in idx:
                self.artist_top_tracks[k].append(v[i])

        self.album_top_tracks = defaultdict(list)
        for k, v in self.album_to_track_id.items():
            track_pops = self.popularity_vec[v]
            idx = np.argsort(1 / track_pops)[0:min(top_k, len(track_pops))].tolist()  # sort artist track by popularity
            for i in idx:
                self.album_top_tracks[k].append(v[i])


    def create_matrices(self, min_track_prior = 0.0002 ):

        self.lsa_track_mask = np.extract(self.popularity_vec > min_track_prior,
                                         np.arange(0,self.popularity_vec.shape[0]))
        self.lsa_id_to_column = dict()
        for i in range(len(self.lsa_track_mask)):
            self.lsa_id_to_column[self.lsa_track_mask[i]] = i


        self.create_train_matrix()
        self.create_test_matrix()
        if self.CHALLENGE_FILE is not None:
            self.create_challenge_matrix()




# END OF CLASS


def load_data(train_size=10000, test_size=2000, load_challenge=False, create_matrices=False, generate_data=False,
              lsa_min_track_prior=0.0002):

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
        d.calculate_popularity()
        if load_challenge:
            print("Load Challenge Set Data")
            d.load_challenge_data()
        if create_matrices:
            print("Calculate Numpy Matrices")
            d.create_matrices(lsa_min_track_prior)

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
    train_size_arg = 10000       # number of playlists for training
    lsa_min_track_prior_arg = 0.0002      # minimum prior probability needed to keep track in LSA training matrix size (default 0.0002 or 2 / 10000 playlists
    test_size_arg = 1000        # number of playlists for testing
    load_challenge_arg = True  # loads challenge data when creating a submission to contest
    create_matrices_arg = True  # creates numpy matrices for train, test, and (possibly) challenge data

    data_in = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg, generate_data_arg, lsa_min_track_prior_arg)

    # lsa = predict_with_LSA()
    # lsa.predict_playlists(svd_components=64)

    pass

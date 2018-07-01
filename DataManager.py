import os
import math
import json
import time
from tqdm import tqdm
import random
from collections import defaultdict

import numpy as np
from sklearn.externals import joblib
from scipy.sparse import lil_matrix, csr_matrix

import re
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


class DataManager:
    """
     Sets up Empty Data Manager object
     Main data structure is three list of lists: playlist ID, track IDs
       where track ID is found in the dictionary uri_to_id and id_to_uri
     Each list represents the three parts of the data set: train, test, challenge
    """

    def __init__(self, path, track_prior, train_size=10000, test_size=2000, challenge_file=None, min_track_prior=0.0):
        self.DATA_DIR = path
        self.CHALLENGE_FILE = challenge_file
        self.track_prior = track_prior
        self.min_track_prior = min_track_prior
        self.data_cache = dict()
        self.train_size = train_size
        self.train = []
        self.train_title = []
        self.train_description = []
        self.word_index_playlist = defaultdict(dict) # token -> playlist -> score
        self.word_index_track = defaultdict(dict)    # token -> track _  ->score



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
        self.track_timestamps = [] #list of modified timestamps for playlists in which the track appears
        self.artist_to_track_id = defaultdict(list)
        self.album_to_track_id = defaultdict(list)
        self.pid_to_spotify_pid = []
        self.X = None
        self.X_test = None
        self.X_test_words = None

        self.X_challenge = None
        self.X_challenge_words = None
        self.popularity_vec = None          # prior probability of track occuring on a playlist

        self.prefix = "spotify:track:"

        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def text_process(self, str):
        str = self.normalize_name(str)
        tokens = word_tokenize(str)
        stemmed_tokens = list()
        for word in tokens:
            if word not in self.stop_words:
                stemmed_tokens.append(self.stemmer.stem(word))
        return stemmed_tokens

    def normalize_name(self, name):
        name = name.lower()
        name = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def add_tokens_to_index(self, index, id, title, description):

        str_lists =[[self.normalize_name(title)], self.text_process(title), self.text_process(description)]
        weights = [1.0, 0.5, 0.25]
        for i in range(len(str_lists)):
            for t in str_lists[i]:
                if t in index.keys():
                    if id in index[t]:
                        index[t][id] += weights[i]
                    else:
                        index[t][id] = weights[i]
                else:
                    index[t] = {id : weights[i]}

    def tfidf_index(self, index, num_docs, mode="ltc"):

        print("Word Index Mode", mode)
        #num_docs = len(index)
        for term in index.keys():

            idf = 1
            if (mode[1] == 't'):
                idf = math.log10(num_docs / len(index[term].keys()))

            for id in index[term]:
                tf = index[term][id]
                if mode[0] == 'l':
                    tf = 1+ math.log10(tf)
                index[term][id] = tf * idf
                if tf*idf < 0:
                    pass

        #length normalization - 2-pass algorithm - sum of squares
        if mode[2] == 'c':
            doc_len = defaultdict(float)
            for term in index.keys():
                for id in index[term].keys():
                    doc_len[id] += index[term][id] ** 2

            for term in index.keys():
                for id in index[term].keys():
                    index[term][id] /= math.sqrt(doc_len[id])

        # check to make sure that each playlist is length 1
        #check_doc_len = defaultdict(float)
        #for term in self.word_index_playlist.keys():
        #    for pid in self.word_index_playlist[term].keys():
        #        check_doc_len[pid] += self.word_index_playlist[term][pid] ** 2
        #pass

    def _add_train_playlist(self, playlist):

        pid = len(self.train)
        self.train.append([])
        title = playlist["name"]
        self.train_title.append(title)
        description = ""
        if "description" in playlist:
            description = playlist["description"]
        self.train_description.append(description)

        self.add_tokens_to_index(self.word_index_playlist, pid, title, description)

        modified = playlist["modified_at"]

        for track in playlist['tracks']:
            track_uri = track['track_uri']
            track_uri = track_uri[len(self.prefix):]

            if self.track_prior[track_uri] < self.min_track_prior:
                continue

            # new track that has never been encountered before
            if track_uri not in self.uri_to_id.keys():
                tid = len(self.id_to_uri)


                self.uri_to_id[track_uri] = tid
                self.id_to_uri[tid] = [track['track_uri'], track['track_name'],
                                       track['artist_uri'], track['artist_name'],
                                       track['album_uri'], track['album_name']]
                self.track_frequency.append(0)
                self.track_timestamps.append(list())

                self.artist_to_track_id[track['artist_uri']].append(tid)
                self.album_to_track_id[track['album_uri']].append(tid)

            track_id = self.uri_to_id[track_uri]
            self.train[pid].append(track_id)
            self.track_frequency[track_id] += 1
            self.track_timestamps[track_id].append(modified)

            self.add_tokens_to_index(self.word_index_track, track_id, title, description)

    def _add_test_playlist(self, playlist):

        subtest = random.randint(0,9)

        # if subtest is already full
        if len(self.test_uri[subtest]) >= self.subtest_size:
            return

        num_tracks = playlist["num_tracks"]

        # not enough tracks to hid any tracks
        # (minimum number of track holdout in challenge data set is 5)
        if num_tracks - 5 <= self.subtest_setup[subtest][1]:
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

    def load_playlist_data(self, mode='ltc'):
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

                is_train = random.uniform(0, 1) > train_test_ratio


                # POTENTIAL DATA LEAKER - Once training is full, everything else can be a test playlist


                # skip playlist if we have already loaded enough of them for either train or test
                if is_train and train_done:
                    is_train = False

                if not is_train and test_done:
                    continue

                if is_train:
                    self._add_train_playlist(playlist)
                    train_done = len(self.train) >= self.train_size
                    if train_done:
                        pass
                    pbar.update(1)

                else:
                    self._add_test_playlist(playlist)
                    test_done = True
                    for i in range(10):
                        if len(self.test_uri[i]) < self.subtest_size:
                            test_done = False
                            break



        pbar.close()
        # TODO: need to explore variants of TF-IDF


        self.tfidf_index(self.word_index_playlist, len(self.train), mode=mode)
        self.tfidf_index(self.word_index_track, len(self.id_to_uri), mode=mode)


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
        # Use file handle to ensure file exists upon serialization
        with open(filename, 'wb') as file:
            joblib.dump(self, file)

    def create_train_matrix(self):
        print(" - train matrix")

        num_rows = len(self.train)
        num_cols = len(self.id_to_uri)

        self.X = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for p in range(num_rows):
            if p % 10000 == 0:
                print(p, " of ", num_rows)
            for t in self.train[p]:
                self.X[p, t] = 1
        self.X = self.X.tocsr()

    def create_test_top_track_matrix(self):
        print(" - test top tracks from artist and album matrix")

        num_subtest = len(self.test)
        num_rows = len(self.test[0])
        num_cols = len(self.id_to_uri)
        self.X_test_top_tracks = list()
        # BUG HERE Make this 0 to num_subtest
        for s in range(0,num_subtest):
            mat = lil_matrix((num_rows, num_cols), dtype=np.int8)

            for p in range(num_rows):
                for track_id in self.test[s][p]:
                    if track_id >= 0:

                        artist_uri = self.id_to_uri[track_id][2]
                        for top_track_id in self.artist_top_tracks[artist_uri]:
                            if track_id != top_track_id:
                                mat[p, top_track_id] = 1

                        album_uri = self.id_to_uri[track_id][4]
                        for top_track_id in self.album_top_tracks[album_uri]:
                            if track_id != top_track_id:
                                mat[p, top_track_id] = 1

            self.X_test_top_tracks.append(mat.tocsc())

    def create_challenge_top_track_matrix(self):
        print(" - challenge top tracks from artist and album matrix")

        num_rows = len(self.challenge)
        num_cols = len(self.id_to_uri)

        mat = lil_matrix((num_rows, num_cols), dtype=np.int8)

        for p in range(num_rows):
            for track_id in self.challenge[p]:
                if track_id >= 0:

                    artist_uri = self.id_to_uri[track_id][2]
                    for top_track_id in self.artist_top_tracks[artist_uri]:
                        if track_id != top_track_id:
                            mat[p, top_track_id] = 1

                    album_uri = self.id_to_uri[track_id][4]
                    for top_track_id in self.album_top_tracks[album_uri]:
                        if track_id != top_track_id:
                            mat[p, top_track_id] = 1

        self.X_challenge_top_tracks= mat.tocsc()

    def create_test_matrix(self):
        print(" - test matrix")

        num_subtest = len(self.test)
        num_rows = len(self.test[0])
        num_cols = len(self.id_to_uri)
        self.X_test = list()
        for s in range(num_subtest):
            mat = lil_matrix((num_rows, num_cols), dtype=np.int8)
            for p in range(num_rows):
                for t in self.test[s][p]:
                    if t >= 0 :
                        mat[p,t] = 1
            self.X_test.append(mat)
        return

    def create_challenge_matrix(self):
        print(" - challenge matrix")
        num_rows = len(self.challenge)
        num_cols = len(self.id_to_uri)
        self.X_challenge = lil_matrix((num_rows, num_cols), dtype=np.int8)
        for p in range(num_rows):
            for t in self.challenge[p]:
                if t >= 0:
                    self.X_challenge[p, t] = 1

    def calculate_popularity(self, top_k = 5):
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

    def create_test_word_matrix_by_playlist_neighbors(self):
        print(" - test title and description word matrix by playlist neighbors:")

        num_subtest = len(self.test)
        num_rows = len(self.test[0])
        num_cols = len(self.id_to_uri)
        self.X_test_words = list()

        pbar = tqdm(total=num_subtest)

        for s in range(0,num_subtest):

            mat = csr_matrix((num_rows, num_cols), dtype="float32")

            for p in range(num_rows):

                tokens  = self.text_process(self.test_title[s][p])
                if len(tokens) > 1:  # add complete title as search token
                    tokens.append(self.normalize_name(self.test_title[s][p]))

                if len(tokens) == 0:
                    continue

                query_token_score = 1/math.sqrt(len(tokens))

                scores = defaultdict(float)
                for token in tokens:
                    if token in self.word_index_playlist.keys():
                        for pid in self.word_index_playlist[token]:
                            scores[pid] += self.word_index_playlist[token][pid] * query_token_score

                #average playlist vectors for all playlists with matching terms

                temp_mat = self.X[list(scores.keys()), :].todense()
                temp_score = np.array(list(scores.values()))
                temp_vec = np.sum(np.multiply(temp_mat.T, temp_score).T, axis=0) /(1+math.log(1+len(scores)))
                # denominator is is used to scale the output so that the maximum value is close to 1
                mat[p, :] = temp_vec

            self.X_test_words.append(mat)
            pbar.update(1)

        print("done.")

    def create_test_word_matrix_by_track_index(self):
        print(" - test title and description word matrix by track index:")

        num_subtest = len(self.test)
        num_rows = len(self.test[0])
        num_cols = len(self.id_to_uri)
        self.X_test_words = list()

        pbar = tqdm(total=num_subtest)

        for s in range(0,num_subtest):

            mat = lil_matrix((num_rows, num_cols), dtype="float32")

            for p in range(num_rows):

                tokens  = self.text_process(self.test_title[s][p])
                if len(tokens) > 1:  # add complete title as search token
                    tokens.append(self.normalize_name(self.test_title[s][p]))

                if len(tokens) == 0:
                    continue

                query_token_score = 1/math.sqrt(len(tokens))

                for token in tokens:
                    if token in self.word_index_track.keys():
                        for tid in self.word_index_track[token]:
                            mat[p,tid] += self.word_index_track[token][tid] * query_token_score

            self.X_test_words.append(mat.tocsr())
            pbar.update(1)

        print("done.")

    def create_challenge_word_matrix_by_playlist_neighbors(self):
        print(" - challenge title and description word matrix")

        num_rows = len(self.challenge)
        num_cols = len(self.id_to_uri)

        mat = csr_matrix((num_rows, num_cols), dtype="float32")

        pbar = tqdm(total=num_rows)
        for p in range(num_rows):
            tokens = self.text_process(self.challenge_title[p])

            query_token_score = 1 / math.sqrt(max(1,len(tokens)))

            scores = defaultdict(float)
            for token in tokens:
                if token in self.word_index_playlist.keys():
                    for pid in self.word_index_playlist[token]:
                        scores[pid] += self.word_index_playlist[token][pid] * query_token_score

            # average playlist vectors for all playlists with matching terms
            temp_mat = self.X[list(scores.keys()), :].todense()
            temp_score = np.array(list(scores.values()))
            temp_vec = np.sum(np.multiply(temp_mat.T, temp_score).T, axis=0) / (1 + math.log(1 + len(scores)))
            # denominator is is used to scale the output so that the maximum value is close to 1
            mat[p, :] = temp_vec
            pbar.update(1)

        pbar.close()
        self.X_challenge_words = mat

    def create_challenge_word_matrix_by_track_index(self):
        print(" - challenge title and description word matrix by track index:")

        num_rows = len(self.challenge)
        num_cols = len(self.id_to_uri)

        mat = lil_matrix((num_rows, num_cols), dtype="float32")
        pbar = tqdm(total=num_rows)
        for p in range(num_rows):

            pbar.update(1)

            # REMOVE LATER: don't compute word matrix for last 5 subchallenges sets
            #if p > 5000:
            #    continue

            tokens  = self.text_process(self.challenge_title[p])
            if len(tokens) > 1:  # add complete title as search token
                tokens.append(self.normalize_name(self.challenge_title[p]))

            if len(tokens) == 0:
                continue

            query_token_score = 1/math.sqrt(len(tokens))

            for token in tokens:
                if token in self.word_index_track.keys():
                    for tid in self.word_index_track[token]:
                        mat[p,tid] += self.word_index_track[token][tid] * query_token_score


        self.X_challenge_words = mat.tocsr()
        pbar.close()
        print("done.")

    def create_matrices(self ):


        self.create_train_matrix()

        self.create_test_matrix()
        #self.create_test_word_matrix_by_playlist_neighbors()
        self.create_test_word_matrix_by_track_index()
        self.create_test_top_track_matrix()

        if self.CHALLENGE_FILE is not None:
            self.create_challenge_matrix()
            #self.create_challenge_word_matrix_by_playlist_neighbors()
            self.create_challenge_word_matrix_by_track_index()
            self.create_challenge_top_track_matrix()


# END OF CLASS

def calculate_track_priors(path, pickle_file):


    prefix = "spotify:track:"
    playlist_count = 0
    track_prior = defaultdict(float)

    for file in os.listdir(path):
        print(file)
        data = json.load(open(path + file))
        for playlist in data['playlists']:
            playlist_count += 1
            for track in playlist['tracks']:
                track_uri = track['track_uri']
                track_uri = track_uri[len(prefix):]
                track_prior[track_uri] += 1.0

    for k in track_prior.keys():
        track_prior[k] /= playlist_count
    joblib.dump(track_prior, pickle_file)
    return track_prior




def load_data(train_size=10000, test_size=2000, load_challenge=False, create_matrices=False, generate_data=False,
              create_pickle_file=True, mode="ltc", min_track_prior= 0.0):

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
        track_prior_pickle_file = pickle_folder + "track_prior.pickle"
        if os.path.isfile(track_prior_pickle_file):
            print("Loading Track Priors")
            track_prior = joblib.load(track_prior_pickle_file)
        else:
            print("Calculating Track Priors")
            track_prior = calculate_track_priors(data_folder, track_prior_pickle_file)

        d = DataManager(data_folder, track_prior, train_size=train_size, test_size=test_size, challenge_file=c_file,
                        min_track_prior=min_track_prior)
        print("Load Playlist Data")
        d.load_playlist_data(mode=mode)
        d.calculate_popularity()
        if load_challenge:
            print("Load Challenge Set Data")
            d.load_challenge_data()
        if create_matrices:
            print("Calculate Numpy Matrices")
            d.create_matrices()

        if create_pickle_file:
            print("Pickle Data into file: "+pickle_file)
            d.pickle_data(pickle_file)
    else:
        print("Load data from Pickle File: "+pickle_file)
        d = joblib.load(pickle_file)
    return d



if __name__ == '__main__':


    generate_data_arg = True      # True - load data for given parameter settings
    #                               False - only load data if pickle file doesn't already exist
    train_size_arg = 1000        # number of playlists for training
    test_size_arg = 1000          # number of playlists for testing
    load_challenge_arg = False    # loads challenge data when creating a submission to contest
    create_matrices_arg = True    # creates numpy matrices for train, test, and (possibly) challenge data
    create_pickle_file_arg = True #takes time to create pickle file
    text_index_mode_arg = "ntc"
    min_track_prior_arg = 0.0002

    data_in = load_data(train_size_arg, test_size_arg, load_challenge_arg, create_matrices_arg,
                        generate_data_arg, create_pickle_file_arg, text_index_mode_arg, min_track_prior_arg)


    pass

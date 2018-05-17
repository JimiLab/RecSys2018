import tensorflow as tf
from DataManager import load_data
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib
import os
from tqdm import tqdm


class LearnDeepEmbeddings:

    def __init__(self, train_size, test_size, embedding_size=256):
        self.data_handle = load_data(train_size, test_size, generate_data=True, create_matrices=True)
        self.embedding_size = embedding_size
        # TODO: Get better value for batch size
        self.batch_size = [10, self.data_handle.X.shape[0]]
        # self.train_set = np.asarray(list())
        # self.target_set = np.asarray(list())
        self.train_set = list()
        self.target_set = list()
        return

    def build_model(self):
        track_embeddings = tf.Variable(tf.random_uniform([self.data_handle.X.shape[1], self.embedding_size], -1.0, 1.0))
        # Placeholders for inputs
        embedding = tf.nn.embedding_lookup(track_embeddings, self.train_set)
        loss = tf.losses.mean_squared_error(labels=self.target_set, )
        return

    def build_training_set(self):
        removal_percent = 0.2
        pbar = tqdm(total=self.data_handle.X.shape[0])
        pbar.write("~~~~~~~ BUILDING TRAINING DATA SET ~~~~~~~")
        for playlist in self.data_handle.X:
            playlist__nonzero_indicies = playlist.nonzero()[1]
            num_to_remove = math.ceil(len(playlist__nonzero_indicies) * removal_percent)
            np.random.shuffle(playlist__nonzero_indicies)
            target_indicies = playlist__nonzero_indicies[0: num_to_remove]
            target = np.zeros(playlist.shape[1])
            target[target_indicies] = 1
            self.target_set.append(target)
            # -1 indicates that the track was removed
            playlist[0, target_indicies] = -1
            self.train_set.append(playlist.toarray().tolist()[0])
            pbar.update(1)
        pbar.close()
        return

    def simple_multilayer_feedforward(self):

        self.train_set = np.asarray(self.train_set)
        self.target_set = np.asarray(self.target_set)
        # self.target_set = to_categorical(self.target_set, num_classes=self.target_set.shape[1])

        model = Sequential()
        model.add(Dense(units=10000, activation="relu", input_dim=self.train_set.shape[1]))
        model.add(Dropout(0.6))
        model.add(Dense(units=1000, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(units=10000, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(units=50000, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(units=100000, activation='relu'))
        model.add(Dropout(0.6))
        model.add(Dense(self.train_set.shape[1], activation='sigmoid'))
        model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=['accuracy'])

        model.fit(self.train_set, self.target_set, epochs=10, batch_size=32, verbose=2)



if __name__ == '__main__':
    train_size_arg = 1000  # number of playlists for training
    test_size_arg = 2000
    l = LearnDeepEmbeddings(train_size_arg, test_size_arg)
    l.build_training_set()
    l.simple_multilayer_feedforward()
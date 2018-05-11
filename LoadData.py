import os
import math
import json
from tqdm import tqdm
import random


class DataManager:

    def __init__(self):
        self.DATA_DIR = os.path.join(os.getcwd(), 'data/mpd.v1/data/')
        self.data_cache = dict()
        self.data_files = os.listdir(self.DATA_DIR)
        self.remove_probability = .1
        return

    # Loads data files sequentially. Should add random option in the future
    # Params:
    #     percentToLoad(float): percentage of data set to load
    #     fields(Array -> String): Array of names of fields in the track object which should be loaded.
    def load_playlists(self, percent_to_load=0.1, fields=["track_uri"], train_split=.66):
        num_playlists = math.ceil(percent_to_load * 1000000)
        num_files_to_load = math.ceil(num_playlists / 1000)
        num_train_playlists = math.ceil(train_split * num_playlists)
        train_set = []
        test_set = {'data': [], 'target': []}
        load_test = False
        pbar = tqdm(total=num_playlists)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')
        for file in self.data_files[:num_files_to_load]:
            data = json.load(open(self.DATA_DIR + file))
            for playlist in data['playlists']:
                num_playlists -= 1
                if load_test:
                    test_set['data'].append({'track_uri': "START OF PLAYLIST"})
                else:
                    train_set.append({'track_uri': "START OF PLAYLIST"})
                for track in playlist['tracks']:
                    if load_test:
                        if self.remove_probability > random.uniform(0, 1):
                            test_set['data'].append('TRACK REMOVED')
                            test_set['target'].append(track['track_uri'])
                        else:
                            test_set['data'].append({k: track[k] for k in fields})
                    else:
                        train_set.append({k: track[k] for k in fields})
                if load_test:
                    test_set['data'].append({'track_uri': 'END OF PLAYLIST'})
                else:
                    train_set.append({'track_uri': 'END OF PLAYLIST'})
                pbar.update(1)
                if not load_test:
                    num_train_playlists -= 1
                if not num_train_playlists:
                    load_test = 1
                if not num_playlists:
                    self.data_cache = {"train": train_set, "test": test_set}
                    pbar.close()
                    return train_set, test_set
        pbar.close()
        self.data_cache = {"train": train_set, "test": test_set}
        return train_set, test_set


if __name__ == '__main__':
    d = DataManager()
    d.load_playlists()

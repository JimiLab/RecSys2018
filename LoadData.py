import os
import math
import json
from tqdm import tqdm
import random
from sklearn.preprocessing import MultiLabelBinarizer

class DataManager:

    def __init__(self):
        self.DATA_DIR = os.path.join(os.getcwd(), 'data/mpd.v1/data/')
        self.dataCache = dict()
        self.datafiles = os.listdir(self.DATA_DIR)
        self.removeProbability = .1
        return

    # Loads data files sequentially. Should add random option in the future
    # Params:
    #     percentToLoad(float): percentage of dataset to load
    #     fields(Array -> String): Array of names of fields in the track object which should be loaded.
    def loadPlaylists(self, percentToLoad=0.1, fields=["track_uri"], trainSplit=.66):
        numPlaylists = math.ceil(percentToLoad * 1000000)
        numFilesToLoad = math.ceil(numPlaylists / 1000)
        numTrainPlaylists = math.ceil(trainSplit * numPlaylists)
        trainSet = []
        testSet = {'data': [], 'target': []}
        loadTest = False
        pbar = tqdm(total=numPlaylists)
        pbar.write('~~~~~~~ LOADING PLAYLIST DATA ~~~~~~~')
        for file in self.datafiles[:numFilesToLoad]:
            data = json.load(open(self.DATA_DIR + file))
            for playlist in data['playlists']:
                numPlaylists -= 1
                if loadTest:
                    testSet['data'].append({'track_uri': "START OF PLAYLIST"})
                else:
                    trainSet.append({'track_uri': "START OF PLAYLIST"})
                for track in playlist['tracks']:
                    if loadTest:
                        if self.removeProbability > random.uniform(0, 1):
                            testSet['data'].append('TRACK REMOVED')
                            testSet['target'].append(track['track_uri'])
                        else:
                            testSet['data'].append({k: track[k] for k in fields})
                    else:
                        trainSet.append({k: track[k] for k in fields})
                if loadTest:
                    testSet['data'].append({'track_uri': 'END OF PLAYLIST'})
                else:
                    trainSet.append({'track_uri': 'END OF PLAYLIST'})
                pbar.update(1)
                if not loadTest:
                    numTrainPlaylists -= 1
                if not numTrainPlaylists:
                    loadTest = 1
                if not numPlaylists:
                    self.dataCache = {"train": trainSet, "test": testSet}
                    pbar.close()
                    return trainSet, testSet
        pbar.close()
        self.dataCache = {"train": trainSet, "test": testSet}
        return trainSet, testSet


if __name__ == '__main__':
    d = DataManager()
    d.loadPlaylists()
from LoadData import DataManager
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from operator import itemgetter
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix

class NaiveModels:

    def __init__(self, compareRandom=False):
        self.dataHandler = DataManager()
        self.train, self.test = self.dataHandler.loadPlaylists(percentToLoad=0.01)
        self.compareRandom = compareRandom
        if compareRandom:
            self.randomBaseline()
        return

    def adjacencyEmbed(self):
        print("~~~~~~~ TRAINING ADJACENCY MODEL ~~~~~~~")
        aM = lil_matrix((self.dataHandler.trackSet.shape[0], self.dataHandler.trackSet.shape[0]))
        trackIndexLookUp = {k: v for v, k in enumerate(self.dataHandler.trackSet)}
        pbar = tqdm(total=self.train.shape[0] + self.test['data'].shape[0])
        for i in range(self.train.shape[0] - 1):
            a1 = trackIndexLookUp[self.train[i]['track_uri']]
            a2 = trackIndexLookUp[self.train[i + 1]['track_uri']]
            aM[a1, a2] += 1
            aM[a2, a1] += 1
            pbar.update(1)
        for i in range(self.test['data'].shape[0] - 1):
            pbar.update(1)
            if self.test['data'][i] != 'TRACK REMOVED' and self.test['data'][i + 1] != 'TRACK REMOVED':
                a1 = trackIndexLookUp[self.test['data'][i]['track_uri']]
                a2 = trackIndexLookUp[self.test['data'][i + 1]['track_uri']]
                aM[a1, a2] += 1
                aM[a2, a1] += 1
        pbar.close()
        print("~~~~~~~ fitting svd ~~~~~~~")
        svd = TruncatedSVD(n_components=2048, algorithm='arpack')
        aM_SVD = svd.fit_transform(aM)
        aM_Similarity = cosine_similarity(aM_SVD, aM_SVD)
        predictions = []
        print("~~~~~~~ generating predictions ~~~~~~~")
        pbar = tqdm(total=self.test['data'].shape[0])
        for i in range(self.test['data'].shape[0]):
            pbar.update(1)
            if self.test['data'][i] == 'TRACK REMOVED':
                if self.test['data'][i - 1] == 'TRACK REMOVED' and self.test['data'][i + 1] == 'TRACK REMOVED':
                    predictions.append("-1")
                elif i == 0 or self.test['data'][i - 1] == 'TRACK REMOVED':
                    next = self.test['data'][i + 1]['track_uri']
                    predictions.append(self.dataHandler.trackSet[aM_Similarity[trackIndexLookUp[next]].argmax()])
                elif i == self.test['data'].shape[0] - 1 or self.test['data'][i + 1] == 'TRACK REMOVED':
                    prev = self.test['data'][i - 1]['track_uri']
                    predictions.append(self.dataHandler.trackSet[aM_Similarity[trackIndexLookUp[prev]].argmax()])
                else:
                    prev = self.test['data'][i - 1]['track_uri']
                    next = self.test['data'][i + 1]['track_uri']
                    if np.amax(aM_Similarity[trackIndexLookUp[prev]]) > np.amax(aM_Similarity[trackIndexLookUp[next]]):
                        predictions.append(self.dataHandler.trackSet[aM_Similarity[trackIndexLookUp[prev]].argmax()])
                    else:
                        predictions.append(self.dataHandler.trackSet[aM_Similarity[trackIndexLookUp[next]].argmax()])
        pbar.close()
        self.evaluate(predictions, self.test['target'], 'ADJACENCY SIMILARITY')
        return

    def bigram(self):
        print("~~~~~~~ TRAINING BIGRAM MODEL ~~~~~~~")
        trackSet = []
        probabilityLookUp = defaultdict(lambda: defaultdict(int))
        conditionCounts = defaultdict(int)
        for i in range(len(self.train) - 1):
            probabilityLookUp[self.train[i]['track_uri']][self.train[i + 1]['track_uri']] += 1
            conditionCounts[self.train[i]['track_uri']] += 1
        for condition, count in conditionCounts.items():
            for event in probabilityLookUp[condition].keys():
                probabilityLookUp[condition][event] /= conditionCounts[condition]
        predictions = []
        for i in range(len(self.test['data']) - 1):
            if self.test['data'][i + 1] == 'TRACK REMOVED':
                    try:
                        predictions.append(max(probabilityLookUp[self.test['data'][i]['track_uri']].items(), key=itemgetter(1))[0])
                    except Exception as e:
                        predictions.append("UNKNOWN")
        self.evaluate(predictions, self.test['target'], 'BIGRAM')
        return

    def randomBaseline(self):
        print("~~~~~~~ TRAINING RANDOM MODEL ~~~~~~~")
        trackSet = list(set([t['track_uri'] for t in self.train]))
        predicitions = []
        for i in range(len(self.test['target'])):
            predicitions.append(random.choice(trackSet))
        self.evaluate(predicitions, self.test['target'], 'RANDOM')
        return predicitions

    def evaluate(self, predictions, truth, modelName):
        print("\n~~~~~~~ EVALUATING " + modelName + " MODEL ~~~~~~~\n")
        print('{0:<20}'.format('ACCURACY'), end="")
        print('{0:<20}'.format('F1'))
        print('{0:<20}'.format(str(round(accuracy_score(truth, predictions), 4))), end="")
        print('{0:<20}'.format(str(round(f1_score(truth, predictions, average='micro'), 4)) + '\n'))
        return

if __name__ == '__main__':
    n = NaiveModels(compareRandom=False)
    # n.adjacencyEmbed()
    n.bigram()
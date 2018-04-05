from LoadData import DataManager
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from operator import itemgetter

class NaiveModels:

    def __init__(self, compareRandom=False):
        self.dataHandler = DataManager()
        self.train, self.test = self.dataHandler.loadPlaylists(percentToLoad=0.2)
        self.compareRandom = compareRandom
        if compareRandom:
            self.randomBaseline()
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
    n = NaiveModels(compareRandom=True)
    n.bigram()
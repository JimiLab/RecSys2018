from LoadData import DataManager
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, f1_score
from operator import itemgetter


class NaiveModels:

    def __init__(self, compare_random=False):
        self.dataHandler = DataManager()
        self.train, self.test = self.dataHandler.loadPlaylists(percentToLoad=0.2)
        self.compareRandom = compare_random
        if compare_random:
            self.randomBaseline()
        return

    def bigram(self):
        print("~~~~~~~ TRAINING BIGRAM MODEL ~~~~~~~")
        probability_look_up = defaultdict(lambda: defaultdict(int))
        condition_counts = defaultdict(int)
        for i in range(len(self.train) - 1):
            probability_look_up[self.train[i]['track_uri']][self.train[i + 1]['track_uri']] += 1
            condition_counts[self.train[i]['track_uri']] += 1
        for condition, count in condition_counts.items():
            for event in probability_look_up[condition].keys():
                probability_look_up[condition][event] /= condition_counts[condition]
        predictions = []
        for i in range(len(self.test['data']) - 1):
            if self.test['data'][i + 1] == 'TRACK REMOVED':
                    try:
                        predictions.append(max(probability_look_up[self.test['data'][i]['track_uri']].items(),
                                               key=itemgetter(1))[0])
                    except KeyError:
                        predictions.append("UNKNOWN")
        self.evaluate(predictions, self.test['target'], 'BIGRAM')
        return

    def random_baseline(self):
        print("~~~~~~~ TRAINING RANDOM MODEL ~~~~~~~")
        track_set = list(set([t['track_uri'] for t in self.train]))
        predicitions = []
        for i in range(len(self.test['target'])):
            predicitions.append(random.choice(track_set))
        self.evaluate(predicitions, self.test['target'], 'RANDOM')
        return predicitions

    @staticmethod
    def evaluate(predictions, truth, model_name):
        print("\n~~~~~~~ EVALUATING " + model_name + " MODEL ~~~~~~~\n")
        print('{0:<20}'.format('ACCURACY'), end="")
        print('{0:<20}'.format('F1'))
        print('{0:<20}'.format(str(round(accuracy_score(truth, predictions), 4))), end="")
        print('{0:<20}'.format(str(round(f1_score(truth, predictions, average='micro'), 4)) + '\n'))
        return


if __name__ == '__main__':
    n = NaiveModels(compareRandom=True)
    n.bigram()

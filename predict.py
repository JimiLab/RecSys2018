import math
from rank_metrics import ndcg_at_k
import numpy as np

def precision_and_recall_at_k(ground_truth, prediction, k=-1):
    """

    :param ground_truth:
    :param prediction:
    :param k: how far down the ranked list we look, set to -1 (default) for all of the predictions
    :return:
    """

    if k == -1:
        k = len(prediction)
    prediction = prediction[0:k]

    numer = len(set(ground_truth).intersection(set(prediction)))
    prec = numer / k
    recall = numer / len(ground_truth)
    return prec, recall


def r_precision(ground_truth, prediction):
    k = len(ground_truth)
    p, r = precision_and_recall_at_k(ground_truth, prediction, k)
    return p


def song_clicks_metric(ranking):
    """
    Spotify p
    :param ranking:
    :return:
    """

    if 1 in ranking:
        first_idx = ranking.index(1)

        return math.floor(first_idx / 10)
    return 51



    @staticmethod
    def print_subtest_results(sub_test_names, metric_names, results):
        (num_subtest, num_metrics) = results.shape
        print('{0: <15}'.format("Subtest"),"\t", end="")
        for i in range(num_metrics):
            print(metric_names[i], "\t", end="")
        print()

        for st in range(num_subtest):
            print('{0: <15}'.format(sub_test_names[st]), "\t", end="")
            for m in range(num_metrics):
                print(np.round(results[st][m],decimals=3), "\t", end="")
            print()

    @staticmethod
    def print_overall_results(metric_names, results):

        print('{0: <15}'.format(""),"\t", end="")
        for i in range(len(metric_names)):
            print(metric_names[i], "\t", end="")
        print()


        print('{0: <15}'.format("Overall"),"\t", end="")
        for m in range(len(metric_names)):
            print(np.round(results[m],decimals=3), "\t", end="")
        print()


if __name__ == '__main__':
    pass

import math
from rank_metrics import ndcg_at_k


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


def ncdg(ranking, k):
    return ndcg_at_k(ranking, k, 0)

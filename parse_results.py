from sklearn.externals import joblib
import numpy as np

[weights, results] = joblib.load("data/pickles/test_results50.pickle")
ncdg = results[:,0,:]

top_score = -1*np.sort(-1*ncdg, axis=1)[:, 0]
top_idx = np.argsort(-1*ncdg, axis=1)[:, 0]

for i in range(top_idx.shape[0]):
    print(i, top_idx[i], top_score[i], weights[top_idx[i]])
    #print(weights[top_idx[i]])
print(np.mean(top_score))
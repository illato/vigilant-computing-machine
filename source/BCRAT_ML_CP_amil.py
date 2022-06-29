import pandas as pd
import numpy as np
import Orange
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import orangecontrib.conformal as cp

import os
#os.chdir(os.path.expanduser('~/conformal_prediction'))
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction')

np.random.seed(42)

df_signal = pd.read_csv('./vigilant-computing-machine/ML_BCP/data/synthetic/signal.csv')


#from sklearn import svm
X = df_signal.iloc[:,1:-1]
y = df_signal.iloc[:,-1]
#clf = svm.SVC()
#clf.fit(X,y)


domain = Orange.data.Domain([ContinuousVariable(f'X{i}') for i in range(9)], DiscreteVariable('Case', map(str, range(2))))
tab = Table(domain, X, y)
normalizer = Orange.preprocess.Normalizer(zero_based = True, norm_type = Orange.preprocess.Normalize.NormalizeBySpan)
ntab = normalizer(tab)


train, test = next(cp.evaluation.RandomSampler(ntab,2,1))
learner = Orange.classification.SVMLearner(probability = True)
#nc = cp.nonconformity.SVMDistance(clf)
nc = cp.nonconformity.InverseProbability(learner)
ic = cp.classification.InductiveClassifier(nc)
r = cp.evaluation.run_train_test(ic, 0.03, train, test)

multi = []
for example, pred, ref in zip(test, r.preds, r.refs):
    if len(pred.classes()) >= 2:
        multi.append((example, pred.classes(), ref))
print(r.accuracy(), len(multi))


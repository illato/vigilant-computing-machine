import numpy as np
import Orange
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import orangecontrib.conformal as cp

import os
#os.chdir(os.path.expanduser('~/conformal_prediction'))
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction/vigilant-computing-machine')

np.random.seed(42)

tab = Orange.data.Table('./data/signal_with_header_for_orange.csv')
normalizer = Orange.preprocess.Normalizer(zero_based = True, norm_type = Orange.preprocess.Normalize.NormalizeBySpan)
ntab = normalizer(tab)


from sklearn import svm
clf = svm.SVC()

train, test = next(cp.evaluation.RandomSampler(ntab,2,1))
nc = cp.nonconformity.SVMDistance(clf)
ic = cp.classification.InductiveClassifier(nc)
r = cp.evaluation.run_train_test(ic, 0.03, train, test)

multi = []
for example, pred, ref in zip(test, r.preds, r.refs):
    if len(pred.classes()) >= 2:
        multi.append((example, pred.classes(), ref))
print(r.accuracy(), len(multi))


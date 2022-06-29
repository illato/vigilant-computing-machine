from copy import deepcopy
import numpy as np
import Orange
import orangecontrib.conformal as cp
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, StringVariable

import os

#os.chdir(os.path.expanduser('~/conformal_prediction'))
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction/vigilant-computing-machine/')
#os.chdir('/uufs/chpc.utah.edu/common/home/uXXXXXXX/conformal_prediction')
#os.chdir('/uufs/chpc.utah.edu/common/home/uXXXXXXX/conformal_prediction')

np.random.seed(42)

# Import modified CSV into Orange Data Table
data = Orange.data.Table("data/signal_with_header_for_orange.csv")
normalizer = Orange.preprocess.Normalizer(zero_based = True, norm_type = Orange.preprocess.Normalize.NormalizeBySpan)
ntab = normalizer(data)

#Split the Train and test dataset
train, test = next(cp.evaluation.RandomSampler(ntab,2,1))

#Call ML Orange SVM, Call conformal prediction tool, train model
learner = Orange.classification.SVMLearner(probability = True)
nc = cp.nonconformity.InverseProbability(learner)
ic = cp.classification.InductiveClassifier(nc)
r = cp.evaluation.run_train_test(ic, 0.03, train, test)

#Print evaluation metrics for model
multi = []
for example, pred, ref in zip(test, r.preds, r.refs):
    if len(pred.classes()) >= 2:
        multi.append((example, pred.classes(), ref))
print(r.accuracy(), len(multi))

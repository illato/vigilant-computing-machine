from copy import deepcopy
import numpy as np
import Orange
import orangecontrib.conformal as cp
from Orange.data import Domain, ContinuousVariable, DiscreteVariable, StringVariable

import os

np.random.seed(42)

#os.chdir(os.path.expanduser('~/conformal_prediction'))
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction/vigilant-computing-machine/')

# Import modified CSV into Orange Data Table
data = Orange.data.Table("data/signal_with_header_for_orange.csv")

# Not a proper train split test but good enough to try Orange functions
train=data[:-1]
test_instance=data[-1]

# Call Orange ML Classier, conformal package, train model
lr = Orange.classification.LogisticRegressionLearner()
ip = cp.nonconformity.InverseProbability(lr)
ccp = cp.classification.CrossClassifier(ip, 5, train)

#Print results
print('Actual class:', test_instance.get_class())
print(ccp(test_instance, 0.1))
print(ccp(test_instance, 0.01))


# 'vigilant-computing-machine/source/util.py'
import util

import numpy as np
from sklearn import svm

#import Orange
import orangecontrib.conformal as cp

import os
# set pwd to root of repository
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction/vigilant-computing-machine')

# set NumPy seed for Orange3-Conformal reproducibility
np.random.seed(42)



tab_random = util.read_csv_to_table('./data/random_with_header_for_orange.csv')
tab_signal = util.read_csv_to_table('./data/signal_with_header_for_orange.csv')
tab_missing = util.read_csv_to_table('./data/missing_with_header_for_orange.csv')
tab_imputed = util.read_csv_to_table('./data/imputed_with_header_for_orange.csv')


def print_result_header():
	print()
	metrics = ['accuracy', 'confidence', 'credibility', 'seconds', 'multiples']
	print('{:>25}'.format(' '), ('{:>12}'*len(metrics)).format(*metrics))


def print_result(r, test, name='cp_or_ncs'):
	multi = []
	for example, pred, ref in zip(test, r.preds, r.refs):
	    if len(pred.classes()) >= 2:
        	multi.append((example, pred.classes(), ref))
	# r.accuracy()
	# r.confidence()
	# r.credibility()
	metrics = [r.accuracy(), r.confidence(), r.credibility(), r.time()]
	# r.confusion(test, pred)
	# r.multiple_criterion()
	# r.singleton_criterion()
	# r.empty_criterion()
	# r.singleton_correct()
	print('{:>25}'.format(name),
	     ('{:>12.3f}'*len(metrics)).format(*metrics),
              '{:>10}'.format(len(multi)))


def run_inductive_svm_distance(clf, tab, name):
	train, test = next(cp.evaluation.RandomSampler(tab,2,1))
	nc = cp.nonconformity.SVMDistance(clf)
	ic = cp.classification.InductiveClassifier(nc)	
	r = cp.evaluation.run_train_test(ic, 0.03, train, test)
	print_result(r, test, name)


def run_inductive_svm_distance_with_pickle(clf, tab, name):
	nc = cp.nonconformity.SVMDistance(clf)
	ic = cp.classification.InductiveClassifier(nc)
	train, test = next(cp.evaluation.RandomSampler(tab,2,1))
	train, calibrate = next(cp.evaluation.RandomSampler(train, 2, 1))
	ic.fit(train, calibrate)
	import pickle
	with open('signal-ic-svmd-svc.pkl', 'wb') as file:
	    pickle.dump(ic, file, pickle.HIGHEST_PROTOCOL)
	ti = test[-1]
	r1 = ic.predict(ti)
	print(r1.confidence(), r1.credibility(), r1.p)
	with open('signal-ic-svmd-svc.pkl', 'rb') as file:
	    test_ic = pickle.load(file)
	r2 = test_ic.predict(ti)
	print(r2.confidence(), r2.credibility(), r2.p)
	#r = cp.evaluation.run_train_test(ic, 0.03, train, test)
	#print_result(r, test, name)


def run_transductive_svm_distance(clf, tab, name):
	train, test = next(cp.evaluation.RandomSampler(tab,2,1))
	nc = cp.nonconformity.SVMDistance(clf)
	tc = cp.classification.TransductiveClassifier(nc)
	r = cp.evaluation.run_train_test(tc, 0.03, train, test)
	print_result(r, test, name)


def run_cross_svm_distance(clf, tab, name):
	train, test = next(cp.evaluation.RandomSampler(tab,2,1))
	nc = cp.nonconformity.SVMDistance(clf)
	cc = cp.classification.CrossClassifier(nc, k = 5)
	r = cp.evaluation.run_train_test(cc, 0.03, train, test)
	print_result(r, test, name)


# confidence  for this NCS appears to be (1 - (p-of-most-probable-other-class))
# credibility for this NCS appears to be (p-of-most-probable-class)
#
# pickle'd model appears* to still be trained
#
#run_inductive_svm_distance_with_pickle(svm.SVC(), tab_random, 'random-IC-SVMDist-SVC')
#run_inductive_svm_distance_with_pickle(svm.SVC(), tab_signal, 'signal-IC-SVMDist-SVC')
#run_inductive_svm_distance_with_pickle(svm.SVC(), tab_missing, 'missing-IC-SVMDist-SVC')
#run_inductive_svm_distance_with_pickle(svm.SVC(), tab_imputed, 'imputed-IC-SVMDist-SVC')
# 0.9626865671641791 0.9664179104477612 [(0.03731343283582089, '0'), (0.9664179104477612, '1')]
# 0.9626865671641791 0.9664179104477612 [(0.03731343283582089, '0'), (0.9664179104477612, '1')]
# 0.9813432835820896 0.5223880597014925 [(0.5223880597014925, '0'), (0.018656716417910446, '1')]
# 0.9813432835820896 0.5223880597014925 [(0.5223880597014925, '0'), (0.018656716417910446, '1')]
# 0.996268656716418 0.9104477611940298 [(0.0037313432835820895, '0'), (0.9104477611940298, '1')]
# 0.996268656716418 0.9104477611940298 [(0.0037313432835820895, '0'), (0.9104477611940298, '1')]
# 0.9626865671641791 0.4962686567164179 [(0.03731343283582089, '0'), (0.4962686567164179, '1')]
# 0.9626865671641791 0.4962686567164179 [(0.03731343283582089, '0'), (0.4962686567164179, '1')]



print_result_header()
run_inductive_svm_distance(svm.LinearSVC(), tab_random, 'random-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_signal, 'signal-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_missing, 'missing-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_imputed, 'imputed-IC-SVMDist-LinSVC')
#
#                              accuracy  confidence credibility     seconds   multiples
# random-IC-SVMDist-LinSVC        0.970       0.768       0.753       1.486        380
# signal-IC-SVMDist-LinSVC        0.943       0.993       0.501       1.117         15
#missing-IC-SVMDist-LinSVC        0.980       0.986       0.514       1.130         54
#imputed-IC-SVMDist-LinSVC        0.985       0.988       0.495       1.118         56


print_result_header()
run_inductive_svm_distance(svm.NuSVC(), tab_random, 'random-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_signal, 'signal-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_missing, 'missing-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_imputed, 'imputed-IC-SVMDist-NuSVC')
#
#                              accuracy  confidence credibility     seconds   multiples
#  random-IC-SVMDist-NuSVC        0.975       0.748       0.764       1.391        379
#  signal-IC-SVMDist-NuSVC        0.985       0.969       0.524       1.319        139
# missing-IC-SVMDist-NuSVC        0.985       0.977       0.527       1.320         91
# imputed-IC-SVMDist-NuSVC        0.940       0.982       0.531       1.318         72


print_result_header()
run_inductive_svm_distance(svm.SVC(), tab_random, 'random-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_signal, 'signal-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_missing, 'missing-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_imputed, 'imputed-IC-SVMDist-SVC')
#
#                              accuracy  confidence credibility     seconds   multiples
#    random-IC-SVMDist-SVC        0.963       0.741       0.739       1.354        367
#    signal-IC-SVMDist-SVC        0.990       0.942       0.589       1.337        216
#   missing-IC-SVMDist-SVC        0.965       0.923       0.551       1.347        257
#   imputed-IC-SVMDist-SVC        0.940       0.958       0.536       1.337        140


print_result_header()
run_cross_svm_distance(svm.SVC(), tab_random, 'random-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_signal, 'signal-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_missing, 'missing-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_imputed, 'imputed-CC-SVMDist-SVC')
#
#                              accuracy  confidence credibility     seconds   multiples
#    random-CC-SVMDist-SVC        0.975       0.767       0.763       3.452        378
#    signal-CC-SVMDist-SVC        0.978       0.966       0.542       3.386        157
#   missing-CC-SVMDist-SVC        0.988       0.938       0.567       3.426        219
#   imputed-CC-SVMDist-SVC        0.970       0.957       0.548       3.376        168


# Changing two occurrences of 'm' in header of CSVs to 'i' fixes below
# error; however, Transductive CP is extremely computationally
# intensive and should not be used when > 100's of examples.
#
# ValueError: Invalid number of meta attribute columns (0 != 2)
#
#print_result_header()
#run_transductive_svm_distance(svm.SVC(), tab_random, 'random-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_signal, 'signal-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_missing, 'missing-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_imputed, 'imputed-TC-SVMDist-SVC')
#
#                              accuracy  confidence credibility     seconds   multiples
#    random-TC-SVMDist-SVC        0.983       0.763       0.731     350.605        381
#    signal-TC-SVMDist-SVC        0.953       0.968       0.521     327.857        144
#   missing-TC-SVMDist-SVC        0.973       0.952       0.537     339.601        181
#   imputed-TC-SVMDist-SVC        0.988       0.964       0.576     329.928        130


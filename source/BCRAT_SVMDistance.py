import numpy as np
from sklearn import svm

import Orange
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
import orangecontrib.conformal as cp

import os
# set pwd to root of repository
os.chdir('/uufs/chpc.utah.edu/common/home/u0740821/conformal_prediction/vigilant-computing-machine')

# set NumPy seed for Orange3-Conformal reproducibility
np.random.seed(42)


def ReadCsvToNormalizedTable(filepath):
	tab = Orange.data.Table(filepath)
	normalizer = Orange.preprocess.Normalizer(
			zero_based = True,
			norm_type = Orange.preprocess.Normalize.NormalizeBySpan)
	return normalizer(tab)


tab_random = ReadCsvToNormalizedTable('./data/random_with_header_for_orange.csv')
tab_signal = ReadCsvToNormalizedTable('./data/signal_with_header_for_orange.csv')
tab_missing = ReadCsvToNormalizedTable('./data/missing_with_header_for_orange.csv')
tab_imputed = ReadCsvToNormalizedTable('./data/imputed_with_header_for_orange.csv')


def print_result_header():
	print()
	metrics = ['accuracy', 'confidence', 'credibility', 'multi']
	print('{:>25}'.format(' '), ('{:>15}'*len(metrics)).format(*metrics))


def print_result(r, test, name='cp_or_ncs'):
	multi = []
	for example, pred, ref in zip(test, r.preds, r.refs):
	    if len(pred.classes()) >= 2:
        	multi.append((example, pred.classes(), ref))
	# r.accuracy()
	# r.confidence()
	# r.credibility()
	metrics = [r.accuracy(), r.confidence(), r.credibility()]
	# r.confusion(test, pred)
	# r.multiple_criterion()
	# r.singleton_criterion()
	# r.empty_criterion()
	# r.singleton_correct()
	print('{:>25}'.format(name),
	     ('{:>15.3f}'*len(metrics)).format(*metrics),
              '{:>14}'.format(len(multi)))



def run_inductive_svm_distance(clf, tab, name):
	train, test = next(cp.evaluation.RandomSampler(tab,2,1))
	nc = cp.nonconformity.SVMDistance(clf)
	ic = cp.classification.InductiveClassifier(nc)
	r = cp.evaluation.run_train_test(ic, 0.03, train, test)
	print_result(r, test, name)


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


print_result_header()
run_inductive_svm_distance(svm.LinearSVC(), tab_random, 'random-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_signal, 'signal-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_missing, 'missing-IC-SVMDist-LinSVC')
run_inductive_svm_distance(svm.LinearSVC(), tab_imputed, 'imputed-IC-SVMDist-LinSVC')

print_result_header()
run_inductive_svm_distance(svm.NuSVC(), tab_random, 'random-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_signal, 'signal-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_missing, 'missing-IC-SVMDist-NuSVC')
run_inductive_svm_distance(svm.NuSVC(), tab_imputed, 'imputed-IC-SVMDist-NuSVC')

print_result_header()
run_inductive_svm_distance(svm.SVC(), tab_random, 'random-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_signal, 'signal-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_missing, 'missing-IC-SVMDist-SVC')
run_inductive_svm_distance(svm.SVC(), tab_imputed, 'imputed-IC-SVMDist-SVC')


print_result_header()
run_cross_svm_distance(svm.SVC(), tab_random, 'random-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_signal, 'signal-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_missing, 'missing-CC-SVMDist-SVC')
run_cross_svm_distance(svm.SVC(), tab_imputed, 'imputed-CC-SVMDist-SVC')
print('')


# TODO:
# ValueError: Invalid number of meta attribute columns (0 != 2)
#
#print_result_header()
#run_transductive_svm_distance(svm.SVC(), tab_random, 'random-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_signal, 'signal-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_missing, 'missing-TC-SVMDist-SVC')
#run_transductive_svm_distance(svm.SVC(), tab_imputed, 'imputed-TC-SVMDist-SVC')
#print()


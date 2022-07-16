import numpy as np
import pandas as pd
from sklearn import svm

import Orange
import orangecontrib.conformal as cp

# set NumPy seed for Orange3-Conformal reproducibility
np.random.seed(42)


def read_csv_to_table(filepath, normalize = True):
    '''
    Reads a CSV with Orange headers and returns an 'Orange.data.Table' that
    is normalized by default

    Parameters
    ----------
    filepath : string
        Path to CSV file.
    normalize : bool, optional
        NormalizeBySpan, else leave as is. The default is True.

    Returns
    -------
    'Orange.data.Table'
        Table of data corresponding to the CSV file (normalized by default).

    '''
    tab = Orange.data.Table(filepath)
    if not normalize:
        return tab
    normalizer = Orange.preprocess.Normalizer(
			zero_based = True,
			norm_type = Orange.preprocess.Normalize.NormalizeBySpan)
    return normalizer(tab)


def pred_to_dict(pred):
    '''
    Takes a conformal classification prediction objects and returns a dict

    Parameters
    ----------
    pred : 'orangecontrib.conformal.classification.PredictionClass'
        Conformal classification prediction object
        produced by 'ConformalClassifier.predict'.

    Returns
    -------
    dict
        dict containing classes, confidence, credibility, eps, and p. 
        Note: 'classes' is numpy.NaN if the prediction object's eps is None.

    '''
    return {
		'classes': np.NaN if pred.eps == None else pred.classes(),
		'confidence': pred.confidence(),
		'credibility': pred.credibility(),
		'eps': pred.eps,
		'p': pred.p
	}


def preds_to_df(preds):
    '''
    Takes a collection of conformal classification prediction objects and 
    returns a pandas DataFrame

    Parameters
    ----------
    preds : Collection of 'orangecontrib.conformal.classification.PredictionClass'
        Collection of conformal classification prediction objects 
        produced by 'ConformalClassifier.predict'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing classes, confidence, credibility, eps, and p.

    '''
    return pd.DataFrame.from_records([pred_to_dict(x) for x in preds])


def result_to_dict(res):
    '''
    Takes a Results object from evaluating a conformal classifier and 
    returns a dict

    Parameters
    ----------
    res : orangecontrib.conformal.evaluation.ResultsClass
        Results from evaluating a conformal classifier.

    Returns
    -------
    dict
        dict containing accuracy, confidence, credibility, singletons, 
        singletons_correct, multiple, and empty..

    '''
    return {
		'accuracy': res.accuracy(),
		'confidence': res.confidence(),		
		'credibility': res.credibility(),
		'singletons': res.singleton_criterion(),
		'singletons_correct': 0 if np.isnan(res.singleton_criterion()) else res.singleton_correct(), ## TODO
		'multiple': res.multiple_criterion(),
		'empty': res.empty_criterion()
	}


def result_to_df(result):
    '''
    Takes a Results object from evaluating a conformal classifier and 
    returns a pandas DataFrame

    Parameters
    ----------
    result : 'orangecontrib.conformal.evaluation.ResultsClass'
        Results from evaluating a conformal classifier.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing accuracy, confidence, credibility, singletons, 
        singletons_correct, multiple, and empty.
        
    '''
    return pd.DataFrame.from_dict(result_to_dict(result))


def set_eps(preds, eps):
    '''
    Set the eps attribute on each prediction in a collection of 
    'orangecontrib.conformal.classification.PredictionClass'

    Parameters
    ----------
    preds : iterable
        Collection of 'orangecontrib.conformal.classification.PredictionClass'.
    eps : float
        Proportion of acceptable error (significance level).
        
    Returns
    -------
    preds : iterable
        Collection of 'orangecontrib.conformal.classification.PredictionClass'.

    '''
    for pred in preds:
        pred.eps = eps
    return preds


def train_test_calibrate_split(in_data,
				train_test_ratio = (2, 1), 
				train_calibrate_ratio = (2, 1)):
    '''
    Takes a table and splits into train, test, and calibration sets.

    Parameters
    ----------
    in_data : 'Orange.data.Table'
        Table of data to be split. Calibration set derives from train set.
    train_test_ratio : 2-tuple, optional
        Train to test ratio. The default is (2, 1).
    train_calibrate_ratio : 2-tuple, optional
        Train to calibrate ratio. The default is (2, 1).

    Returns
    -------
    train : iterable
        Train data.
    test : iterable
        Test data.
    calibrate : iterable
        Calibrate data.

    '''
    train, test = next(cp.evaluation.RandomSampler(in_data,
                                                   train_test_ratio[0],
                                                   train_test_ratio[1]))
    train, calibrate = next(cp.evaluation.RandomSampler(train,
                                                        train_calibrate_ratio[0],
                                                        train_calibrate_ratio[1]))
    return train, test, calibrate


def train_calibrate_split(in_data, train_calibrate_ratio = (2, 1)):
    '''
    Takes a table and splits into train and calibration sets.

    Parameters
    ----------
    in_data : 'Orange.data.Table'
        Table of data to be split. Calibration set derives from train set.
    train_calibrate_ratio : 2-tuple, optional
        Train to calibrate ratio. The default is (2, 1).

    Returns
    -------
    train : iterable
        Train data.
    calibrate : iterable
        Calibrate data.

    '''
    train, calibrate = next(cp.evaluation.RandomSampler(in_data,
                                                        train_calibrate_ratio[0],
                                                        train_calibrate_ratio[1]))
    return train, calibrate


def fit_inductive_inverse_probability(in_learner,
                                      in_data, 
                                      mondrian = True):
    '''
    Wraps in_learner in an inductive conformal predictor that uses the 
    inverse probability nonconformity measure, which is then fit to in_data.

    Parameters
    ----------
    in_learner : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    in_data : 'Orange.data.Table'
        Table of data to run.
    mondrian : bool, optional
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes. The default is True.

    Returns
    -------
    'orangecontrib.conformal.classification.InductiveClassifier'
        Trained/calibrated inductive conformal prediction classifier.

    '''
    train, calibrate = train_calibrate_split(in_data)
    return fit_inductive_inverse_probability_split(in_learner, train, calibrate)

def fit_inductive_inverse_probability_split(in_learner, 
                                            train, 
                                            calibrate,
                                            mondrian = True):
    '''
    Wraps in_learner in an inductive conformal predictor that uses the 
    inverse probability nonconformity measure, which is then fit to in_data.

    Parameters
    ----------
    in_learner : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    train : 'Orange.data.Table'
        Table of train data.
    calibrate : 'Orange.data.Table'
        Table of calibrate data.
    mondrian : bool, optional
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes. The default is True.

    Returns
    -------
    'orangecontrib.conformal.classification.InductiveClassifier'
        Trained/calibrated inductive conformal prediction classifier.

    '''
    ic = inductive_inverse_probability(in_learner, mondrian)
    ic.fit(train, calibrate)
    return ic


def run_inductive_inverse_probability(in_learner, in_data, mondrian = True, eps = 0.05):
    '''
    Wraps in_learner in an inductive conformal predictor that uses the 
    inverse probability nonconformity measure and trains/calibrates the 
    combination on in_data--the results of which are returned.

    Parameters
    ----------
    in_learner : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    in_data : 'Orange.data.Table'
        Table of data to run.
    mondrian : bool, optional
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes. The default is True.
    eps : float, optional
        Proportion of acceptable error (significance level). 
        The default is 0.05.

    Returns
    -------
    'orangecontrib.conformal.evaluation.ResultsClass'
        Results from evaluating a conformal classifier.

    '''
    train, test, calibrate = train_test_calibrate_split(in_data)
    return run_inductive_inverse_probability_split(in_learner,
												   mondrian,
												   train,
												   test,
												   calibrate,
												   eps)


def run_inductive_inverse_probability_split(in_learner, mondrian, train, test, calibrate = None, eps = 0.05):
	ic = inductive_inverse_probability(in_learner, mondrian)
	return cp.evaluation.run_train_test(ic, eps, train, test, calibrate)


#def run_inductive_probability_margin(in_learner, in_data, eps = 0.05):
#def run_inductive_probability_margin_split(in_learner, train, test, calibrate, eps = 0.05):

#def run_inductive_svm_distance(svm, in_data, eps = 0.05):
#def run_inductive_svm_distance_split(svm, train, test, calibrate, eps = 0.05):

def run_inductive_LOOC(in_learner, in_data, eps = 0.05):
	train, test, calibrate = train_test_calibrate_split(in_data)
	return run_inductive_LOOC_split(in_learner,
					train, 
					test, 
					calibrate, 
					eps)

def run_inductive_LOOC_split(in_learner, train, test, calibrate, eps = 0.05):
	nc = cp.nonconformity.LOOClassNC(in_learner)
	ic = cp.classification.InductiveClassifier(nc)
	return cp.evaluation.run_train_test(ic, eps, train, test, calibrate)


#def run_inductive_knn_distance(in_learner, in_data, eps = 0.05):
#def run_inductive_knn_distance_split(in_learner, train, test, calibrate, eps = 0.05):

#def run_inductive_knn_fraction(in_learner, in_data, eps = 0.05):
#def run_inductive_knn_fraction_split(in_learner, train, test, calibrate, eps = 0.05):

    
def inductive_inverse_probability(clf, mondrian=True):
    '''
    Creates an inductive conformal prediction classifier that produces
    inverse probability nonconformity measures

    Parameters
    ----------
    clf : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    mondrian : bool, optional
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes. The default is True.

    Returns
    -------
    ic : 'orangecontrib.conformal.classification.InductiveClassifier'
        Inductive conformal prediction classifier.

    '''
    nc = cp.nonconformity.InverseProbability(clf)
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    return ic


def inductive_probability_margin(clf):
	nc = cp.nonconformity.ProbabiltiyMargin(clf)
	return cp.classification.InductiveClassifier(nc)


def inductive_svm_distance(clf = svm.SVC()):
	nc = cp.nonconformity.SVMDistance(clf)
	return cp.classification.InductiveClassifier(nc)


def inductive_looc(clf, 
				   dist = Orange.distance.Euclidean(), 
				   k = 10, 
				   relative = True, 
				   include = False, 
				   neighbourhood = 'fixed'):
	nc = cp.nonconformity.LOOClassNC(clf, dist, k, relative, include, neighbourhood)
	return cp.classification.InductiveClassifier(nc)


def inductive_knn_distance(clf = Orange.distance.Euclidean(), 
						   k = 1):
	nc = cp.nonconformity.KNNDistance(clf, k)
	return cp.classification.InductiveClassifier(nc)


def inductive_knn_fraction(clf = Orange.distance.Euclidean(), 
				   k = 1, 
				   weighted = False):
	nc = cp.nonconformity.KNNFraction(clf, k, weighted)
	return cp.classification.InductiveClassifier(nc)
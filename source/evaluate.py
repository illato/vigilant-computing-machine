
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import orangecontrib.conformal as cp

import util

# set NumPy seed for Orange3-Conformal reproducibility
np.random.seed(42)



def run_inductive_inverse_probability(in_learner, mondrian, train, test, calibrate = None, eps = 0.05):
    '''
    Wraps in_learner in an inductive conformal predictor that uses the 
    inverse probability nonconformity measure and trains/calibrates/tests the 
    combination on and returns the results.

    Parameters
    ----------
    in_learner : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    mondrian : bool
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes.
    train : 'Orange.data.Table'
        Table containing train data.
    test : 'Orange.data.Table'
        Table containing test data.
    calibrate : 'Orange.data.Table', optional
        Table containing calibration data. The default is None (train:calib = 2:1).
    eps : float, optional
        Proportion of acceptable error (significance level). 
        The default is 0.05.

    Returns
    -------
    'orangecontrib.conformal.evaluation.ResultsClass'
        Results from evaluating a conformal classifier.

    '''
    ic = util.inductive_inverse_probability(in_learner, mondrian)
    return cp.evaluation.run_train_test(ic, eps, train, test, calibrate)


def make_graph_inductive_inverse_probability(clf, tab, ratio_train_test=[1,4], eps=0.05, mondrian=True):
    '''
    Produces scaled and unscaled graphs of confidence/credibility/accuracy of 
    the results of clf in mondrian InverseProbability-InductiveClassifier.

    Parameters
    ----------
    clf : 'Orange.base.Learner'
        Classifier/Learner that implements 'Orange.base.Learner' or similar.
    tab : 'Orange.data.Table'
        Table of data to run.
    ratio_train_test : list, optional
        Ratio of train:test (train:calibrate = 2:1, from train). The default is [1,4].
    eps : float, optional
        Proportion of acceptable error (significance level). 
        The default is 0.05.
    mondrian : bool
        Mondrian conformal prediction calculates separate nonconformity scores 
        for each class, which may increase score accuracy of 
        less observed classes.

    Returns
    -------
    fig : figure
        unscaled.
    fig_min_max : figure
        scaled.

    '''
    train, test = next(cp.evaluation.RandomSampler(tab, a=ratio_train_test[1], b=ratio_train_test[0]))
    train, calibrate = next(cp.evaluation.RandomSampler(train, 2, 1))
    nc = cp.nonconformity.InverseProbability(clf)
    ic = cp.classification.InductiveClassifier(nc, mondrian=True)
    ic.fit(train, calibrate)
    results = cp.evaluation.run_train_test(ic, 
                                           eps, 
                                           train, 
                                           test, 
                                           calibrate)
    model_preds = results.preds    
    df_model_preds = util.preds_to_df(model_preds)
    scaler = MinMaxScaler()
    df_model_preds ['instance'] = test
    df_model_preds ['accuracy'] = results.accuracy()
    df_model_preds[['confidence', 'credibility']] = df_model_preds[['confidence', 'credibility']]
    df_model_preds=df_model_preds.sort_values(['confidence', 'credibility'], 
                                              ascending=[False, False]).reset_index(drop=True)

    fig, ax = plt.subplots()
    df_model_preds.plot(y=['confidence'], use_index=True, ax=ax, linewidth=2.5)
    df_model_preds.plot(y=['credibility','accuracy'], use_index=True, ax=ax)
    fig.suptitle('%s' % str(clf).title() , fontsize=16)
    plt.vlines(x = [np.quantile(list(df_model_preds.index), .25),
                    np.quantile(list(df_model_preds.index), .50),
                    np.quantile(list(df_model_preds.index), .75)], 
               ymin = 0, 
               ymax = 1,
               colors = 'purple',
               linestyles= 'dashed')
    fig=fig


    df_model_preds[['confidence', 'credibility']] = \
        scaler.fit_transform(df_model_preds[['confidence', 'credibility']])
    df_model_preds=df_model_preds.sort_values(['confidence', 'credibility'], 
                                              ascending=[False, False]).reset_index(drop=True)

    fig2, ax = plt.subplots()
    df_model_preds.plot(y=['confidence'], use_index=True, ax=ax, linewidth=2.5)
    df_model_preds.plot(y=['credibility','accuracy'], use_index=True, ax=ax)
    ax.set_yticks(ticks=[0,1]) 
    ax.set_yticklabels(labels=['Min','Max'])
    fig2.suptitle('%s as Min Max' % str(clf).title() , fontsize=16)
    plt.vlines(x = [np.quantile(list(df_model_preds.index), .25),
                    np.quantile(list(df_model_preds.index), .50),
                    np.quantile(list(df_model_preds.index), .75)], 
               ymin = 0, 
               ymax = 1,
               colors = 'purple',
               linestyles= 'dashed')
    fig_min_max=fig2
    return fig, fig_min_max



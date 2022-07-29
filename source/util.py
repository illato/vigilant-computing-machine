
import numpy as np
import pandas as pd

import Orange
import orangecontrib.conformal as cp

# set NumPy seed for Orange3-Conformal reproducibility
np.random.seed(42)

# confidence  in binary classification is (1 - (p-of-most-probable-other-class); probability the opposite outcome is not observed)
# credibility in binary classification is (p-of-most-probable-class)


def sort_reindex(df_pred, col=['confidence','credibility']):
    '''
    Sort df by the provided column(s), update index to sorted order

    Parameters
    ----------
    df_pred : 'pandas.DataFrame'
        A dataframe to be sorted by the provided columns.
    col : list, optional
        Column(s) to sort from left to right. 
        The default is ['confidence','credibility'].

    Returns
    -------
    'pandas.DataFrame'
        The sorted and reindexed dataframe.

    '''
    if type(col) == str:
        return df_pred.sort_values(by=col, ascending=False).reset_index(drop=True)
    if type(col) == list:
        return df_pred.sort_values(by=col, ascending=[False]*len(col)).reset_index(drop=True)


def plot_confidence_credibility_by_index(df_pred, title='', ax=None, alpha=1.0):
    '''
    Plot 'confidence' and 'credibility' columns of dataframe by index

    Parameters
    ----------
    df_pred : 'pandas.DataFrame'
        dataframe to plot.
    title : string, optional
        plot title. The default is ''.
    ax : 'matplotlib.pyplot.axis', optional
        axis to plot on. The default is None.
    alpha : float, optional
        alpha value of lines. The default is 1.0.

    Returns
    -------
    None.

    '''
    df_pred.plot(y=['confidence', 'credibility'], use_index=True, title=title, ax=ax, alpha=alpha)
    
def plot_conf_cred(df_pred, title='', scale=True, ax=None, alpha=1.0):
    '''
    Plot df sorted and indexed by 'confidence' and 'credibility'

    Parameters
    ----------
    df_pred : 'pandas.DataFrame'
        dataframe to plot.
    title : string, optional
        plot title. The default is ''.
    scale : bool, optional
        scale plot. The default is True.
    ax : 'matplotlib.pyplot.axis', optional
        axis to plot on. The default is None.
    alpha : float, optional
        alpha value of lines. The default is 1.0.

    Returns
    -------
    None.

    '''
    df = sort_reindex(df_pred, ['confidence', 'credibility'])
    if scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[['confidence', 'credibility']] = scaler.fit_transform(df[['confidence', 'credibility']])
    plot_confidence_credibility_by_index(df, title=title, ax=ax, alpha=alpha)


def tab_x_col_to_int_arr(tab, col_idx):
    '''
    Coerce an 'Orange.data.Table' column to int and return as a 'NumPy.array'

    Parameters
    ----------
    tab : 'Orange.data.Table'
        table containing column to extract.
    col_idx : int
        index of column to extract.

    Returns
    -------
    'NumPy.array'
        array containing table's column values coerced to int.

    '''
    return np.apply_along_axis(lambda x: int(x), axis=0, arr=tab.X[:,col_idx].reshape(1,-1))


def tab_x_col_to_arr(tab, col_idx):
    '''
    Extract an 'Orange.data.Table' column and return as a 'NumPy.array'

    Parameters
    ----------
    tab : 'Orange.data.Table'
        table containing column to extract.
    col_idx : int
        index of column to extract.

    Returns
    -------
    'NumPy.array'
        array containing table's column values.

    '''
    return tab.X[:,col_idx]


def df_col_to_int_if_float_is_integer_all(df, col_name):
    '''
    Coerce df column to int if all are float where float.is_integer == True

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing column to potentially coerce to int.
    col_name : string
        name of column.

    Returns
    -------
    None.

    '''
    # if all instances in this column are integers in float representation
    if df.loc[:,col_name].apply(float.is_integer).all():
        # convert all instances in this column to integer
        df.loc[:,col_name] = df.loc[:,col_name].astype(int)


def add_insts_to_pred_dfs(dfs, test):
    '''
    Add test instances to prediction dfs

    Parameters
    ----------
    dfs : list of 'pandas.DataFrame'
        list of dfs containing predictions.
    test : List of 'Orange.data.Table'
        List of tables containing test instances.

    Returns
    -------
    out_dfs : list of 'pandas.DataFrame'
        List of dataframes with a new column corresponding to the test instances.

    '''
    out_dfs = []
    for i in range(len(dfs)):
        # get df, shallow-copy, do not mutate state
        df = dfs[i].copy()
        # get corresponding Orange Table of test instances
        tab = test[i]
        # get X attributes (to get column names)
        attrs = tab.domain.attributes
        for col_idx, col_name in [(i,x.name) for i,x in zip(list(range(len(attrs))), attrs)]:
            # for each X variable, add as column in df
            df.loc[:,col_name] = tab_x_col_to_arr(tab, col_idx=col_idx)
            # if all integers represented as floats, convert to int
            df_col_to_int_if_float_is_integer_all(df, col_name)
        # add y as column representing ground truth class of each instance
        df.loc[:,'class'] = tab.Y
        df_col_to_int_if_float_is_integer_all(df, 'class')
        out_dfs.append(df)
    return out_dfs


def df_pred_empty_mean(df, pred_col_name='classes'):
    '''
    Calculate the proportion of predictions 
    (generally, could be any column containing a type that can have len() applied) 
    that are empty.

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing predictions.
    pred_col_name : string, optional
        column name. The default is 'classes'.

    Returns
    -------
    float
        proportion of empty entries.

    '''
    return df[pred_col_name].apply(lambda x: len(x) < 1).mean()


def df_pred_singleton_mean(df, pred_col_name='classes'):
    '''
    Calculate the proportion of predictions 
    (generally, could be any column containing a type that can have len() applied) 
    that are of length 1.

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing predictions.
    pred_col_name : string, optional
        column name. The default is 'classes'.

    Returns
    -------
    float
        proportion of singleton entries.

    '''
    return df[pred_col_name].apply(lambda x: len(x)==1).mean()


def df_pred_singleton_correct_mean(df, pred_col_name='classes', verdict_col_name='verdict'):
    '''
    Calculate the proportion of predictions that are of length 1 and contain 
    the correct class.

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing predictions and verdicts.
    pred_col_name : string, optional
        prediction column name. The default is 'classes'.
    verdict_col_name : string, optional
        verdict column name. The default is 'verdict'.


    Returns
    -------
    float
        proportion of singleton entries that correspond to the correct class.

    '''
    single_and_correct = (df[pred_col_name].apply(lambda x: len(x)==1) & df[verdict_col_name])
    return sum(single_and_correct) / len(df)


def df_pred_multiple_mean(df, pred_col_name='classes'):
    '''
    

    Calculate the proportion of predictions 
    (generally, could be any column containing a type that can have len() applied) 
    that are of length greater than 1.

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing predictions.
    pred_col_name : string, optional
        column name. The default is 'classes'.

    Returns
    -------
    float
        proportion of multiple entries.

    '''
    return df[pred_col_name].apply(lambda x: len(x) > 1).mean()


def df_pred_inst_metrics(df, dataset_name='Ming', mondrian=False, method='', model_for_setup=None):
    '''
    Calculate metrics and add as columns to a dataframe that already contains 
    columns for prediction attributes. Provide optional parameters to increase 
    detail on setup that produced predictions, generally for comparison to other 
    setups.

    Parameters
    ----------
    df : 'pandas.DataFrame'
        dataframe containing prediction attributes as columns.
    dataset_name : string, optional
        name of dataset used. The default is 'Ming'.
    mondrian : bool, optional
        True if the conformal prediction classifer was mondrian. The default is False.
    method : string, optional
        The ML method, approach, and/or other relevant data. The default is ''.
    model_for_setup : model, optional
        one instance of the model, used to capture attribtes. The default is None.

    Returns
    -------
    s : 'pandas.DataFrame'
        DataFrame containing prediction attributes, metrics, and setup details.

    '''
    s = df[['confidence','credibility','verdict']].mean()
    s['empty']=df_pred_empty_mean(df)
    s['single']=df_pred_singleton_mean(df)
    s['single_correct']=df_pred_singleton_correct_mean(df)
    s['multiple']=df_pred_multiple_mean(df)
    s['data'] = dataset_name
    s['mondrian'] = mondrian
    s['classifier'] = model_for_setup.nc_measure.__dict__['model'].name if model_for_setup != None else ''
    s['conformal_predictor'] = model_for_setup.__format__('') if model_for_setup != None else ''
    s['nonconformity'] = str(model_for_setup.nc_measure) if model_for_setup != None else ''
    s['method'] = method
    s['model'] = str(model_for_setup.nc_measure.__dict__['model']) if model_for_setup != None else ''
    return s


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


def table_to_df(tab, x_only=True):
    '''
    Creates a pandas DataFrame from an Orange Table. Excludes y by default.

    Parameters
    ----------
    tab : 'Orange.data.Table'
        'Orange.data.Table' containing instances/rows of X,y data.
    x_only : bool, optional
        Only include X data in the DataFrame, not y data. The default is True.

    Returns
    -------
    df : 'pandas.DataFrame'
        DataFrame containing data from the Orange Table.

    '''
    n = len(tab.domain.attributes)
    names = []
    for i in range(n):
        names.append(tab.domain.attributes[i].name)
    names
    df = pd.DataFrame(tab.X[:,range(0,n)])
    df.columns = names # ['T1','N_Biop', 'HypPlas', 'AgeMen','Age1st', 'N_Rels', 'Race']
    if x_only:
        return df
    df[tab.domain.class_var.name] = tab.Y
    return df


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
		'singletons_correct': 0 if np.isnan(res.singleton_criterion()) else res.singleton_correct(),
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
        Note: eps < 0 will result in all classes being predicted; 
        eps > 1 will result in zero classes being predicted
        
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
    Takes a table and splits into train, test, and calibration sets (RandomSampler).

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
    Takes a table and splits into train and calibration sets (RandomSampler).

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
    return fit_inductive_inverse_probability_split(in_learner, train, calibrate, mondrian)

    
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


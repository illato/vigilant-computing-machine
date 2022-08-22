
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Orange
import orangecontrib.conformal as cp



def run_experiments(cp_in, train, test, calib, eps=0.1, times=1, method=''):
    from copy import deepcopy
    preds_dfs = []
    for run in range(times):
        cc = deepcopy(cp_in)
        cc.fit(train, calib)
        preds = []
        verds = []
        for ex in test:
            pred = cc.predict(ex, eps)
            verds.append(
                pred.verdict(ex.get_class().value, 
                             eps))
            preds.append(pred)
        preds_df = preds_to_df(preds)
        preds_df['verdict'] = verds
        add_inst_to_pred_df(preds_df, 
                            test)
        preds_dfs.append(
            df_pred_inst_metrics(preds_df, 
                                 dataset_name=test.name,
                                 mondrian=cc.mondrian,
                                 method=method, 
                                 model_for_setup=cc))
    return pd.concat(preds_dfs, axis=1).T


def run_unsplit_experiment(inductive_cp, table, normalizer, out_dir='./', 
                           name='unsplit_experiment', eps=0.1):
    from pathlib import PurePath
    import os
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tab = normalizer(table)
    df_ex = run_experiments(cp_in=inductive_cp, train=tab, test=tab, calib=tab, 
                            eps=eps, times=1, method=name)
    df_meta = df_ex.iloc[:, :-1]
    df_meta['normalizer'] = str(normalizer.__dict__)
    df_meta.to_csv(PurePath(out_dir, f'{name}_experiment.csv'), index_label=False)
    df = df_ex.loc[0, 'df']
    for i,meta_attr in enumerate(table.domain.metas):
        df[meta_attr] = tab.metas[:, i]
    df.to_csv(PurePath(out_dir, f'{name}_predictions.csv'), index_label=False)


def create_class_imbalance(tab):
    # get non-cancer
    non_cancer_indices = np.array([x.row_index for x in tab if x.get_class().value == '0'])

    # get cancer (randomized, but reproducible)
    cancer_indices = np.array([x.row_index for x in tab if x.get_class().value == '1'])
    cancer_indices_randomized = np.random.default_rng(42).permutation(cancer_indices)

    # get 1 in 10 cancer
    cancer_indices_tenth = cancer_indices_randomized[-int(len(cancer_indices) / 10):]

    # create new table with healthy:cancer == 10:1
    return Orange.data.Table.from_table_rows(tab, 
                                             np.concatenate([cancer_indices_tenth, 
                                                             non_cancer_indices]))


def plot_confidence_by_verdict(df, ax=None, title_suffix=''):
    import seaborn as sns
    df = sort_reindex(df)
    cnt = max(df.index) - min(df.index)
    g = sns.lineplot(x=range(cnt+1), y='confidence', hue='verdict', data=df, alpha=0.4, linewidth=4, ax=ax)
    g.set_title(f'Confidence by Verdict\n(true class in prediction set){title_suffix}')
    ldr_max_conf = get_low_confidence_predictions(df, percentile=10).confidence.max()
    g.hlines(ldr_max_conf, xmin=0, xmax=cnt, color='r', linestyles='--', label='LDR')
    g.tick_params(axis='x', bottom=False, labelbottom=True)
    g.legend()


def plot_credibility_by_verdict(df, ax=None, title_suffix=''):
    import seaborn as sns
    df = sort_reindex(df)
    cnt = max(df.index) - min(df.index)
    g = sns.lineplot(x=range(cnt+1), y='credibility', hue='verdict', data=df, alpha=0.4, linewidth=4, ax=ax)
    g.set_title(f'Credibility by Verdict\n(true class in prediction set){title_suffix}')
    ldr_max_cred = get_low_credibility_predictions(df, percentile=10).credibility.max()
    g.hlines(ldr_max_cred, xmin=0, xmax=cnt, color='r', linestyles='--', label='LDR')
    g.tick_params(axis='x', bottom=False, labelbottom=True)
    g.legend()


def plot_race_representation_all_vs_low(all_rep, low_rep, clf, data_name, mondrian, sample_size, ax=None):
    width = 0.35
    x = np.arange(len(all_rep.index))

    if ax is None:
        fig, ax = plt.subplots()
        fig.tight_layout()
        fig.set_figwidth(10)
        fig.set_figheight(10)

    ax.bar(x - width/2, 
           all_rep, 
           width, 
           color='blue', 
           label='population')

    ax.bar(x + width/2, 
           low_rep, 
           width, 
           color='orange', 
           label='low confidence')

    ax.legend()
    ms = (lambda x: ' (Mondrian)' if x else '')
    ax.set_title(f'Race Representation - Population Sample vs Low Confidence\n{clf}\n{data_name}\nn={sample_size}{ms(mondrian)}')

    
def read_experiment(experiment_csv):
    pred_csv = experiment_csv.replace('experiment', 'predictions')
    exp = pd.read_csv(experiment_csv)
    df = pd.read_csv(pred_csv)
    df.classes = df.classes.apply((lambda x: list(x.replace('\"','')
                                                  .replace('[','')
                                                  .replace('\'','')
                                                  .replace(',','')
                                                  .replace(' ','')
                                                  .replace(']',''))))
    exp['df'] = [df]
    return exp    

    
def plot_race_representation_from_experiment(exp, ax=None, treat_empty_as_low_conf=False):
    mondrian = exp.loc[0, 'mondrian']
    cpred = exp.loc[0, 'conformal_predictor']
    data_name = exp.loc[0, 'data']
    df = exp.loc[0, 'df']
    sample_size = len(df)

    n = df.Race.value_counts().sort_index()
    nr = n.values.sum()
    d = pd.DataFrame()
    d['all_race_n'] = n
    d['all_race'] = n / nr

    lc = get_low_confidence_predictions(df, 10)
    if treat_empty_as_low_conf:
        emp = df[get_empty_predictions_mask(df)]
        lc = df.loc[lc.index.join(emp.index, how='outer')]
    n = lc.Race.value_counts().sort_index()
    nr = n.values.sum()
    d['low_race_n'] = n
    d['low_race'] = n / nr
    d = d.fillna(0)
    d['low_race_n'] = d['low_race_n'].astype(int)
    plot_race_representation_all_vs_low(d.all_race, d.low_race.values, 
                                        cpred, data_name, mondrian, sample_size, ax=ax)
    d


def plot_experiments(df_experiments, data_title=None, scale=True):
    import matplotlib.pyplot as plt
    
    is_mondrian = (lambda x: ' (Mondrian)' if x else '')
    is_scaled = (lambda x: '\n(Confidence/Credibility Scaled)' if x else '')
    fig, ax = plt.subplots()

    for i, run in df_experiments.iterrows():    
        clf = run.classifier if run.classifier is not None else 'KNN'
        plot_conf_cred(run.df, 
                       title=f'{clf}{is_mondrian(run.mondrian)}\ndata: {run.data if data_title is None else data_title}{is_scaled(scale)}', 
                       ax=ax,
                       scale=scale,
                       alpha=0.5)
        ax.hlines(y=run.verdict, 
                  xmin=0, 
                  xmax=len(run.df), 
                  label=f'acc {run.verdict*100:.2f}%', 
                  color='g', 
                  linestyles='--')
        ax.vlines(x=len(run.df)/4, ymin=0, ymax=1, color='k', lw=3, linestyles=(0, (1, 10)))
        ax.vlines(x=len(run.df)/2, ymin=0, ymax=1, color='k', lw=3, linestyles=(0, (1, 10)))
        ax.vlines(x=len(run.df)*3/4, ymin=0, ymax=1, color='k', lw=3, linestyles=(0, (1, 10)))
        ax.legend()


def plot_prediction(preds_list_s, min_conf=None, min_cred=None, title=None):
    fig,ax = plt.subplots()
    
    confs = []
    creds = []
    label_x = []
    label_y = []
    error_x = []
    error_y = []
    for i,pred_series in enumerate(preds_list_s):
        confs.append(pred_series.confidence * -1)
        creds.append(pred_series.credibility)
        classes = [int(c) for c in pred_series.classes]
        true_class = int(pred_series.loc['class'])
        label_x.append(i)
        label_y.append(true_class)
        if true_class not in classes:
            error_x.append(i)
            if true_class == 0:
                error_y.append(1)
            else:
                error_y.append(0)
    
    x = list(range(len(confs)))
    ax.scatter(x, creds, marker='o', color='b', clip_on=False, label='Cred', zorder=2)
    ax.scatter(x, confs, marker='o', color='g', clip_on=False, label='Conf', zorder=2)
    ax.scatter(error_x, error_y, marker='X', color='r', clip_on=False, label='Error', zorder=3, s=120)
    ax.scatter(label_x, label_y, marker='o', color='k', clip_on=False, label='Label', zorder=4)
    
    ax.legend(bbox_to_anchor=(1.204, 1), loc='upper right')

    markerline, stemlines, baseline = plt.stem(range(len(creds)), creds)
    conf_stemlines = stemlines
    plt.setp(baseline, 'color', 'k')
    plt.setp(baseline, 'linewidth', 0.25)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(stemlines, 'linewidth', 4)
    plt.setp(markerline, 'color', 'b')
    markerline, stemlines, baseline = plt.stem(range(len(confs)), confs)
    plt.setp(baseline, 'color', 'k')
    plt.setp(baseline, 'linewidth', 0.25)
    plt.setp(stemlines, 'color', 'g')
    plt.setp(stemlines, 'linewidth', 4)
    plt.setp(markerline, 'color', 'g')
    cred_stemlines = stemlines
    
    ax.scatter(error_x, error_y, marker='x', color='k', clip_on=False, label='Error', zorder=3, s=100, linewidth=0.5)
    if min_cred is not None:
        ax.hlines(min_cred, xmin=0, xmax=len(confs)-1, linewidth=1.5, linestyle='--', zorder=5, color='r')
    if min_conf is not None:
        ax.hlines(-min_conf, xmin=0, xmax=len(confs)-1, linewidth=1.5, linestyle='--', zorder=5, color='r')

    plt.ylim((-1,1))
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_ylabel('Probability measure\n(Confidence plotted as -Confidence),\n0/1 class label (error)')
    ax.set_xlabel('Conformal Predictors')
    if title is not None:
        ax.set_title(title)
    [spine.set_linewidth(0.25) for edge,spine in ax.spines.items()]

    plt.show()


def get_low_confidence_predictions(df, percentile=10):
    #check valid range
    assert 0 < percentile and percentile < 100, 'percentile outside valid range'
    if percentile < 1:
        percentile = percentile * 100
    #integer math rounds down, preventing rounding up beyond bounds for small collections (though .head() handles)
    #open to reasons to round up
    num_to_return = len(df) // percentile
    #when predictions with the same confidence value span the percentile boundary,
    #credibility will be used as the secondary sort criteria
    #when predictions with the same confidence and credibility span the percentile boundary,
    #predictions will be returned based upon their index values in the provided DataFrame
    df = df.sort_values(by=['confidence', 'credibility'], kind='mergesort')
    return df.iloc[:num_to_return, :]


def get_low_credibility_predictions(df, percentile=10):
    #check valid range
    assert 0 < percentile and percentile < 100, 'percentile outside valid range'
    if percentile < 1:
        percentile = percentile * 100
    #integer math rounds down, preventing rounding up beyond bounds for small collections (though .head() handles)
    #open to reasons to round up
    num_to_return = len(df) // percentile
    #when predictions with the same credibility value span the percentile boundary,
    #confidence will be used as the secondary sort criteria
    #when predictions with the same confidence and credibility span the percentile boundary,
    #predictions will be returned based upon their index values in the provided DataFrame
    df = df.sort_values(by=['credibility', 'confidence'], kind='mergesort')
    return df.iloc[:num_to_return, :]


def get_experiments_max_low_confidence(experiments):
    return [get_low_confidence_predictions(exp.df[0], percentile=10).confidence.max() for exp in experiments]


def get_experiments_max_low_credibility(experiments):
    return [get_low_credibility_predictions(exp.df[0], percentile=10).credibility.max() for exp in experiments]


def get_experiments_median_low_credibility(experiments):
    return [get_low_credibility_predictions(exp.df[0], percentile=10).credibility.median() for exp in experiments]


def get_incorrect_predictions(df):
    return df[~df['verdict']]

def get_empty_predictions_mask(df):
    return df.classes.apply((lambda x: len(x) == 0))

def get_single_predictions_mask(df):
    return df.classes.apply((lambda x: len(x) == 1))

def get_multiple_predictions_mask(df):
    return df.classes.apply((lambda x: len(x) > 1))


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
        return df_pred.sort_values(by=col, ascending=False, kind='mergesort').reset_index(drop=True)
    if type(col) == list:
        return df_pred.sort_values(by=col, ascending=[False]*len(col), kind='mergesort').reset_index(drop=True)


def plot_confidence_credibility_by_index(df_pred, title='', ax=None, alpha=1.0, legend=True):
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
    legend : bool, optional
        display legend in graph. The default is True

    Returns
    -------
    None.

    '''
    df_pred.plot(y=['confidence', 'credibility'], use_index=True, title=title, ax=ax, alpha=alpha, legend=legend)
    
def plot_conf_cred(df_pred, title='', scale=True, ax=None, alpha=1.0, legend=True):
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
    legend : bool, optional
        display legend in graph. The default is True

    Returns
    -------
    None.

    '''
    df = sort_reindex(df_pred, ['confidence', 'credibility'])
    if scale:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[['confidence', 'credibility']] = scaler.fit_transform(df[['confidence', 'credibility']])
    plot_confidence_credibility_by_index(df, title=title, ax=ax, alpha=alpha, legend=legend)


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
        add_inst_to_pred_df(df, tab)
        out_dfs.append(df)
    return out_dfs


def add_inst_to_pred_df(df, test):
    '''
    Add test instances to prediction df

    Parameters
    ----------
    df : 'pandas.DataFrame'
        df containing predictions.
    test : 'Orange.data.Table'
        Table containing test instances.

    Returns
    -------
    None.

    '''
    tab = test
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
    if model_for_setup is not None:
        try:        
            nc = model_for_setup.nc_measure
            mod = model_for_setup.nc_measure.__dict__['model']
            if isinstance(nc, cp.nonconformity.ClassModelNC) and mod is not None:
                s['classifier'] = mod.name
                s['model'] = str(mod)
            elif isinstance(nc, cp.nonconformity.ClassNearestNeighboursNC) and mod is not None:
                s['classifier'] = f'KNN - {type(model_for_setup.nc_measure.distance).__name__}'
                s['model'] = ''
            else:
                s['classifier'] = f'{type(model_for_setup).__name__} - {type(model_for_setup.nc_measure).__name__}'
                s['model'] = ''
        except KeyError:
            s['classifier'] = ''
            s['model'] = ''
        s['conformal_predictor'] = model_for_setup.__format__('')
        s['nonconformity'] = str(model_for_setup.nc_measure)
        s['method'] = method
        s['instance_of_model'] = model_for_setup
    else:
        s['classifier'] = ''
        s['conformal_predictor'] = ''
        s['nonconformity'] = ''
        s['method'] = method
        s['instance_of_model'] = model_for_setup if model_for_setup != None else ''
    s['df'] = df
    return s


def read_csv_to_table(filepath):
    '''
    Reads a CSV with Orange headers and returns an 'Orange.data.Table'.

    Parameters
    ----------
    filepath : string
        Path to CSV file.

    Returns
    -------
    'Orange.data.Table'
        Table of data corresponding to the CSV file.

    '''
    return Orange.data.Table(filepath)


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
    df = pd.DataFrame()
    for i, attribute in enumerate(tab.domain.attributes):
        df[attribute.name] = tab.X[:, i]

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


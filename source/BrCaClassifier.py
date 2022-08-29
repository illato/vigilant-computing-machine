
import util

import pandas as pd

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter

base = importr('base')
utils = importr('utils')

# set seed for reproducibility
base.set_seed(42)

# TODO rpy2 usings

# Note: R (R_HOME) needs to be in PATH


class BrCaClassifier():
    
    def __init__(self, train_in=None):
        self.bcra = importr('BCRA')
        
        # if R package 'BCRA' is not installed
        if not rpackages.isinstalled('BCRA'):
            # selecting first mirror in list for R packages prevents selection dialog
            utils.chooseCRANmirror(ind=1)
            # install R package 'BCRA'
            utils.install_packages('BCRA')
            
        assert rpackages.isinstalled('BCRA')
        bcra = importr('BCRA')
        self.bcra_version = bcra.__version__ # 2.1.2
        
        if type(train_in) != type(None):
            self.fit(train_in)
            
            
    #def __call__(self, instance):
        #return self.predict(instance)
    def __call__(self, train_in=None):
        if type(train_in) != type(None):
            return BrCaClassifier(train_in)
        
            
    def fit(self, train_in):
        if type(train_in) == Orange.data.Table:
            self.domain = train_in.domain
            train_in = util.table_to_df(train_in, x_only=False)
            if 'ID' not in train_in.columns.values:
                train_in.index.name = 'ID'
                train_in = train_in.reset_index()
        assert (type(train_in) == pd.DataFrame)
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.train = self.coerce_df_dtypes(train_in.copy())
            self.train['absolute_risk'] = self.bcra.absolute_risk(self.train)
            self.median_abs_risk = self.train.absolute_risk.median()
            self.train['pred'] = self.train['absolute_risk'] >= self.median_abs_risk

            
    def predict(self, pd_df):
        assert self.median_abs_risk > 0 # has already been fit
        if type(pd_df) == Orange.data.RowInstance:
            pd_df = Orange.data.Table.from_list(pd_df.domain, [pd_df])
        if type(pd_df) == Orange.data.Table:
            pd_df = util.table_to_df(pd_df, x_only=False)
            if 'ID' not in pd_df.columns.values:
                pd_df.index.name = 'ID'
                pd_df = pd_df.reset_index()
        with localconverter(ro.default_converter + pandas2ri.converter):
            ar = self.bcra.absolute_risk(ro.DataFrame(pd_df))
            return ar >= self.median_abs_risk


    def coerce_df_dtypes(self, df_in):
        df_out = pd.DataFrame(df_in).copy()
        df_out = df_out.astype({
            'ID': int,
            'T1': float,
            'T2': float,
            'N_Biop': int,
            'HypPlas': int,
            'AgeMen': int,
            'Age1st': int,
            'N_Rels': int,
            'Race': int
        })
        return df_out


from orangecontrib.conformal.nonconformity import ClassNC
class BrCaNonconformity(ClassNC):


    def __init__(self, classifier):
        """Store the provided classifier as :py:attr:`learner`."""
        assert  isinstance(classifier, BrCaClassifier)
        #self.learner = classifier
        self.clf = classifier
        self.model = None


    def fit(self, data):
        #assert len(data.domain.class_var.values) == 2, \
        #    "BrCaNonconformity only supports binary classification"
        self.clf.fit(data)


    def nonconformity(self, instance):
        y = self.clf.predict(instance)
        ar = self.clf.bcra.absolute_risk(instance)
        df = self.clf.train
        aar = df.loc[(df['pred']==y), 'absolute_risk'] 
        return sum(aar >= ar) / len(aar)


#from orangecontrib.conformal.nonconformity import ClassModelNC
#class BrCaNonconformity(ClassModelNC):


#    def __init__(self, classifier):
#        """Store the provided classifier as :py:attr:`learner`."""
#        assert  isinstance(classifier, BrCaClassifier)
#        self.learner = classifier
#        self.model = None

import Orange
import orangecontrib.conformal as cp

import os
os.chdir('C:/Users/Bob/CHPC/conformal_prediction/vigilant-computing-machine/source')
tab_sig = Orange.data.Table('../ML_BCP/data/synthetic/signal.csv')
tab = Orange.data.Table('../data/signal_with_header_for_orange_include_T2.csv')
train, test = next(cp.evaluation.RandomSampler(tab, 2, 1))
train, calibrate = next(cp.evaluation.RandomSampler(train, 2, 1))
ic = cp.classification.InductiveClassifier(
    BrCaNonconformity(
        BrCaClassifier()),
        train, 
        calibrate)
pred = ic.predict(test)

ic = cp.classification.InductiveClassifier(
    cp.nonconformity.InverseProbability(
        Orange.classification.LogisticRegressionLearner()), 
    train, 
    calibrate)
pred = ic.predict(test)


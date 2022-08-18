import source.util as util

import Orange
import orangecontrib.conformal as cp

is_mondrian = (lambda x: '_mondrian' if x else '')
is_balanced = (lambda x: '_balanced' if x else '_imbalanced')


def run_experiment_logistic_regression_1200(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_1200{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_logistic_regression_12000(mondrian=False, balanced=True):
    data_name = '10x Ming et al. - 1:1 - healthy:cancer' if balanced else \
                '10x Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_12000{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_knn_1200(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_1200{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_12000(mondrian=False, balanced=True):
    data_name = '10x Ming et al. - 1:1 - healthy:cancer' if balanced else \
                '10x Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_12000{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)    

    
def run_experiment_knn_fraction_1200(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_1200{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_knn_fraction_12000(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_12000{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_1200(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_1200{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_12000(mondrian=False, balanced=True):
    data_name = '10x Ming et al. - 1:1 - healthy:cancer' if balanced else \
                '10x Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_12000{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_rf_1200(mondrian=False, balanced=True):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_1200{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_rf_12000(mondrian=False, balanced=True):
    data_name = '10x Ming et al. - 1:1 - healthy:cancer' if balanced else \
                '10x Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_12000{is_balanced(balanced)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)

import source.util as util
import source.classification as classification

import Orange
import orangecontrib.conformal as cp

is_mondrian = (lambda x: '_mondrian' if x else '')
is_balanced = (lambda x: '_balanced' if x else '_imbalanced')
is_race_conditional = (lambda x: '_race-conditional' if x else '')

def run_experiment_logistic_regression_1200(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'logistic_regression_1200{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_1200_race_unique(mondrian=False, balanced=True, race_conditional=False):
    data_name = 'Race-Unique Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Unique Risk - 10:1 - healthy:cancer'
    path = './data/signal_race_unique.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_1200_race-unique{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_1200_race_unique_inverted(mondrian=False, balanced=True, race_conditional=False):
    data_name = 'Race-Unique Risk Inverted - 1:1 - healthy:cancer' if balanced else \
                'Race-Unique Risk Inverted - 10:1 - healthy:cancer'
    path = './data/signal_race_unique_inverted.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_1200_race-unique-inverted{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_1200_rrr(mondrian=False, balanced=True):
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_1200_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_logistic_regression_1200_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_1200_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_knn_1200(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'knn_1200{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)    


def run_experiment_knn_1200_rrr(mondrian=False, balanced=True):
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_1200_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_1200_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_1200_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)

    
def run_experiment_knn_fraction_1200(mondrian=False, balanced=True, race_conditional=False):
    data_name = 'Ming et al. - 1:1 - healthy:cancer' if balanced else \
                'Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_1200{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_fraction_1200_rrr(mondrian=False, balanced=True):
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_1200_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_fraction_1200_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_1200_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)    
    
    
def run_experiment_ada_1200(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'ada_1200{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_1200_rrr(mondrian=False, balanced=True):
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_1200_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_1200_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_1200_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_rf_1200(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'rf_1200{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_rf_1200_rrr(mondrian=False, balanced=True):
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_1200_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_rf_1200_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = 'Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                'Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_1200_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_logistic_regression_12000(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'logistic_regression_12000{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_12000_race_unique(mondrian=False, balanced=True, race_conditional=False):
    data_name = '10x Race-Unique Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Unique Risk - 10:1 - healthy:cancer'
    path = './data/signal_race_unique_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_12000_race-unique{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_12000_race_unique_inverted(mondrian=False, balanced=True, race_conditional=False):
    data_name = '10x Race-Unique Risk Inverted - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Unique Risk Inverted - 10:1 - healthy:cancer'
    path = './data/signal_race_unique_inverted_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_12000_race-unique-inverted{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_logistic_regression_12000_rrr(mondrian=False, balanced=True):
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_12000_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_logistic_regression_12000_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.LogisticRegressionLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'logistic_regression_12000_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_12000(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'knn_12000{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)


def run_experiment_knn_12000_rrr(mondrian=False, balanced=True):
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_12000_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_12000_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.KNNLearner()
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_12000_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_knn_fraction_12000(mondrian=False, balanced=True, race_conditional=False):
    data_name = '10x Ming et al. - 1:1 - healthy:cancer' if balanced else \
                '10x Ming et al. - 10:1 - healthy:cancer'
    path = './data/signal_10x_with_header_for_orange.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_12000{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_fraction_12000_rrr(mondrian=False, balanced=True):
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_12000_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_knn_fraction_12000_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1
    nc = cp.nonconformity.KNNFraction(distance=Orange.distance.distance.Euclidean(), k=5)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'knn_fraction_12000_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps) 

    
def run_experiment_ada_12000(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'ada_12000{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_12000_rrr(mondrian=False, balanced=True):
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_12000_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_ada_12000_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.modelling.ada_boost.SklAdaBoostClassificationLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'ada_12000_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    

def run_experiment_rf_12000(mondrian=False, balanced=True, race_conditional=False):
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
    experiment_name = f'rf_12000{is_balanced(balanced)}{is_race_conditional(race_conditional)}{is_mondrian(mondrian)}'
    
    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian) \
        if not race_conditional else \
        classification.RaceConditionalInductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_rf_12000_rrr(mondrian=False, balanced=True):
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_12000_rrr{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = cp.classification.InductiveClassifier(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
    
    
def run_experiment_rf_12000_rrr_rcic(mondrian=False, balanced=True):
    """Deprecated. Note: this experiment is only retained to preserve the logic that produced corresponding
    experiment results. RaceConditionalIndClf creates an indvidual underlying classifier for each race, which
    means that no one classifier has seen the entire distribution provided to fit--invalidating assumptions.
    """
    data_name = '10x Race-Rel. Risk - 1:1 - healthy:cancer' if balanced else \
                '10x Race-Rel. Risk - 10:1 - healthy:cancer'
    path = './data/signal_non_linear_rrr_10x.csv'
    out_dir = './results/'
    eps = 0.1    
    clf = Orange.classification.RandomForestLearner(random_state=42)
    nc = cp.nonconformity.InverseProbability(clf)

    tab = Orange.data.Table(path)
    tab = tab if balanced else util.create_class_imbalance(tab)
    tab.name = data_name
    experiment_name = f'rf_12000_rrr_rcic{is_balanced(balanced)}{is_mondrian(mondrian)}'

    norm = Orange.preprocess.Normalizer()
    ic = classification.RaceConditionalIndClf(nc, mondrian=mondrian)
    util.run_unsplit_experiment(ic, tab, norm, out_dir=out_dir, name=experiment_name, eps=eps)
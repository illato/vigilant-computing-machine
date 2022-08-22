from orangecontrib.conformal.classification import InductiveClassifier
from copy import deepcopy
import numpy as np

class RaceConditionalIndClf(InductiveClassifier):
    """Race-Conditional Inductive classification.
    Attributes:
        alpha: Nonconformity scores of the calibration instances. Computed by the :py:func:`fit` method.
    """

    def __init__(self, nc_measure, mondrian=False):
        """Initialize feature-conditional inductive classifier based on Race with a nonconformity measure.
        Args:
            nc_measure (ClassNC): Classification nonconformity measure.
            mondrian (bool): Use a mondrian setting for computing label-conditional p-values.
        """
        super().__init__(nc_measure, mondrian=mondrian)
        self.nc_measure_base = deepcopy(self.nc_measure)
        self.mondrian_feature = 'Race'

    def fit(self, train, calibrate):
        """Fit the conformal classifier to the training set, compute and store nonconformity scores (:py:attr:`alpha`)
        on the calibration set and store the domain.
        Args:
            train: Table of examples used as a training set.
            calibrate: Table of examples used as a calibration set.
        """
        self.domain = train.domain
        self.calibrate = calibrate
        self.mf_idx = None
        mf = 'Race'
        for i,attr in enumerate(train.domain.attributes):
            if attr.name == mf:
                self.mf_idx = i
                break
        assert self.mf_idx is not None
        self.ics = {}
        mf_values_zero_based = np.array(self.domain[self.mf_idx].values).astype(float) - 1
        for race in mf_values_zero_based:
            race_train = train[(np.array(train[:, self.mf_idx])==race).flatten(), :]
            race_calibrate = calibrate[(np.array(calibrate[:, self.mf_idx])==race).flatten(), :]
            train_classes = np.unique(race_train.Y)
            train_class_cnt = len(train_classes)
            try:
                if train_class_cnt < 2:
                    import warnings
                    warnings.warn(f'Feature-Conditional Inductive Classifier needs at least 2 classes in the train data, but the data only contains: {train_classes}. Instances of Race {race} will always have p-values of 0!')
                    self.ics[race] = None
                else:
                    self.ics[race] = InductiveClassifier(deepcopy(self.nc_measure_base), 
                                                         race_train, 
                                                         race_calibrate, 
                                                         mondrian=self.mondrian)
            except:
                import warnings
                if str(self.nc_measure_base.__dict__['learner']) == 'knn':
                    warnings.warn(f'KNN learner needs at least k examples of Race {race} in the train data. Race {race} will always have p-values of 0!')
                else:
                    warnings.warn(f'An error occurred, likely involving the number of examples of Race {race} in the train data. Race {race} will always have p-values of 0!')
                self.ics[race] = None
                    

    def p_values(self, example):
        """Compute feature-conditional (Race) p-values for every possible class.
        Inductive classifier assigns an assumed class value to the given example and compares its nonconformity
        against all other instances in the calibration set.
        Args:
            example (Instance): Orange row instance.
        Returns:
            List of pairs (p-value, class)
        """
        ex = np.array(example).flatten()
        race_ic = self.ics.get(ex[self.mf_idx], None)
        if race_ic is None:
            return [(0, y) for y in example.domain.class_var.values]
        return race_ic.p_values(example)

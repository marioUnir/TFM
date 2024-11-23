
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture

class OutliersDetection:
    """
    Parameters
    ----------
    method : str (default='iqr')
        Method to detect outliers. Use `'esd'` for generalized extreme student
        deviate test or use `'iqr'` for inter quantile range test.

    k : float (default=1.5)
        Multiplication value to the IQR that set the outliers labelling
        limits:

        .. math::

            \\text{lower_limit} = Q_1 - k * (Q_3 - Q_1)\\\\
            \\text{upper_limit} = Q_3 + k * (Q_3 - Q_1)

    r_max : float (default=0.01)
        Maximum percentage of records to be tested as true outliers.

    max_pvalue : float (default=0.05)
        The maximum p-value used in the tests.

    n_samples : int
        Number of records in the data.

    relative_error : float or int (default=1e-4)
        The relative target precision to achieve (>= 0). If set to zero,
        the exact percentiles are computed, which could be very expensive.
        Applies only in the `distributed` flavour.
    """

    def __init__(
            self,
            method,
            k=1.5,
            seed=42,
            estimator=None,
            contamination=0.05,
            n_neighbors=20,
            n_components=2
    ):

        self.method = method
        self.k = k
        self.seed = seed
        self.estimator = estimator
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        # Initialize
        self.numerical_features = None
        self.dict_column = {}

        self._check_parameters()

    def _check_parameters(self):

        if self.method not in ["IQR", "std", "IForest", "LOF", "GMM"]:
            raise ValueError("`method` should be 'IQR', 'std', 'IForest', 'LOF', or 'GMM'.")

        if not isinstance(self.k, (float, int)) and self.method in ["IQR", "std"]:
            raise TypeError("`k` must be a number for 'IQR' or 'std'.")

    def run(self, data):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        self.numerical_features = self._get_numerical_features(data)

        self._detect_outliers(data)

    def _get_numerical_features(self, data):

        numerical_vars = data.select_dtypes(
            include=['number']).columns.tolist()

        numerical_vars = [var for var in numerical_vars if data[var].nunique() > 2]

        return numerical_vars

    def _detect_outliers(self, data):
        """
        Detect outliers based on the selected method.
        """

        if self.method == "IQR":
            self._detect_outliers_iqr(data)
        elif self.method == "std":
            self._detect_outliers_std(data)
        elif self.method in ["IForest", "LOF", "GMM"]:
            self._initialize_estimator()
            if self.method in ["IForest", "LOF"]:
                self._detect_outliers_iforest_lof(data)
            else:
                self._detect_outliers_gmm(data)

    def _detect_outliers_iqr(self, data):
        """
        Detect outliers using the Interquartile Range (IQR) method.
        """
        for var in self.numerical_features:
            Q1 = data[var].quantile(0.25)
            Q3 = data[var].quantile(0.75)
            IQR = Q3 - Q1

            lb = Q1 - self.k * IQR
            ub = Q3 + self.k * IQR

            outliers = data[(data[var] < lb) | (data[var] > ub)].index.tolist()
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            self.dict_column[var] = {
                "Q1": Q1,
                "Q3": Q3,
                "IQR": IQR,
                "lb": lb,
                "ub": ub,
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_std(self, data):
        """
        Detect outliers using the standard deviation method.
        """
        for var in self.numerical_features:
            mean = data[var].mean()
            std_dev = data[var].std()

            lb = mean - self.k * std_dev
            ub = mean + self.k * std_dev

            outliers = data[(data[var] < lb) | (data[var] > ub)].index.tolist()
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            self.dict_column[var] = {
                "mean": mean,
                "std_dev": std_dev,
                "lb": lb,
                "ub": ub,
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _initialize_estimator(self):

        if self.estimator is None:
            if self.method == "IForest":
                self.estimator = IsolationForest(contamination=self.contamination, random_state=self.seed)
            elif self.method == "LOF":
                self.estimator = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
            elif self.method == "GMM":
                self.estimator = GaussianMixture(n_components=self.n_components, random_state=self.seed)

    def _detect_outliers_iforest_lof(self, data):

        for var in self.numerical_features:
            data_reshaped = data[var].values.reshape(-1, 1)
            labels = self.estimator.fit_predict(data_reshaped)
            outliers = np.where(labels == -1)[0]
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            self.dict_column[var] = {
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_gmm(self, data):

        for var in self.numerical_features:
            data_reshaped = data[var].values.reshape(-1, 1)
            self.estimator.fit(data_reshaped)
            scores = self.estimator.score_samples(data_reshaped)
            threshold = np.percentile(scores, 5)
            outliers = np.where(scores < threshold)[0]
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            self.dict_column[var] = {
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def get_outliers(self, variable):
        """
        Get the list of outliers for a specific variable.

        Parameters
        ----------
        variable : str
            The name of the variable.

        Returns
        -------
        list
            List of indices of the outliers.
        """
        if variable not in self.dict_column:
            raise ValueError(f"Variable {variable} not analyzed.")
        return self.dict_column[variable]["outliers"]

    def get_results(self):
        """
        Provide a summary of outliers for all numerical variables.

        Returns
        -------
        pandas.DataFrame
            Summary of outlier detection for each variable.
        """

        return pd.DataFrame(self.dict_column).T

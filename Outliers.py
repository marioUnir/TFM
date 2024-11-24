
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture


class OutliersDetection:
    """
    Parameters
    ----------
    method : str
        Method to detect outliers.

    k : float (default=1.5)
        Multiplication value to the IQR that set the outliers labelling
        limits:

        If method is IQR, then

        .. math::

            \\text{lower_limit} = Q_1 - k * (Q_3 - Q_1)\\\\
            \\text{upper_limit} = Q_3 + k * (Q_3 - Q_1)

        If method is std, then

        .. math::

            \\text{lower_limit} = mean - k * std\\\\
            \\text{upper_limit} = mean + k * std

    threshold : float (default=0.05)
        Threshold to identify outliers in categorical features
    """

    def __init__(
            self,
            method,
            k=1.5,
            threshold=0.05,
            seed=42,
            estimator=None,
            contamination=0.05,
            n_neighbors=20,
            n_components=2
    ):

        self.method = method
        self.k = k
        self.threshold = threshold
        self.seed = seed
        self.estimator = estimator
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        # Initialize
        self.numerical_features = None
        self.categorical_features = None
        self.dict_column = {}
        self.dict_column_cat = {}

        self._check_parameters()

    def _check_parameters(self):

        if self.method not in ["IQR", "std", "IForest", "LOF", "GMM", "all"]:
            raise ValueError("`method` should be 'IQR', 'std', 'IForest', 'LOF', 'GMM' or 'all'.")

        if not isinstance(self.k, (float, int)) and self.method in ["IQR", "std"]:
            raise TypeError("`k` must be a number for 'IQR' or 'std'.")

        if self.estimator is not None:
            if isinstance(self.estimator, IsolationForest):
                self.method = "IForest"
            elif isinstance(self.estimator, LocalOutlierFactor):
                self.method = "LOF"
            elif isinstance(self.estimator, GaussianMixture):
                self.method = "GMM"
            else:
                raise ValueError("`estimator` should be a IsolationForest, LocalOutlierFactor or GaussianMixture "
                                 "estimator of sklearn")

    def run(self, data):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        self.numerical_features, self.categorical_features = self._get_numerical_categorical_features(data)

        self._detect_outliers(data)

    def _get_numerical_categorical_features(self, data):

        numerical_vars = data.select_dtypes(
            include=['number']).columns.tolist()
        categorical_vars = data.select_dtypes(
            exclude=['number']).columns.tolist()

        # Add variables explicitly if some numerical values are categorical (e.g., Gender, Marriage)
        categorical_vars += [
            col for col in numerical_vars if data[col].nunique() < 10
        ]
        numerical_vars = [
            col for col in numerical_vars if col not in categorical_vars
        ]

        return numerical_vars, categorical_vars

    def _initialize_estimator(self):

        if self.estimator is None:
            if self.method == "IForest":
                self.estimator = IsolationForest(contamination=self.contamination, random_state=self.seed)
            elif self.method == "LOF":
                self.estimator = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
            elif self.method == "GMM":
                self.estimator = GaussianMixture(n_components=self.n_components, random_state=self.seed)

    def _detect_outliers(self, data):
        """
        Detect outliers based on the selected method.
        """

        print("** Numerical Features **")
        self._detect_outliers_numerical(data)

        print("** Categorical Features **")
        self._detect_outliers_categorical(data)

    def _detect_outliers_numerical(self, data):
        if self.method == "all":
            self.dict_column = {}
            all_methods = ["IQR", "std", "IForest", "LOF", "GMM"]
            for method in all_methods:
                print(f"Running {method} method...")
                self.method = method
                self._initialize_estimator()  # Reinitialize estimator if needed
                self.dict_column[self.method] = {}
                if method in ["IForest", "LOF", "GMM"]:
                    self._detect_outliers_iforest_lof(data) if method != "GMM" else self._detect_outliers_gmm(data)
                else:
                    getattr(self, f"_detect_outliers_{method.lower()}")(data)
            self.method = "all"  # Reset method to 'all'
        else:
            self.dict_column[self.method] = {}
            if self.method in ["IForest", "LOF", "GMM"]:
                self._initialize_estimator()
                if self.method in ["IForest", "LOF"]:
                    self._detect_outliers_iforest_lof(data)
                else:
                    self._detect_outliers_gmm(data)
            else:
                getattr(self, f"_detect_outliers_{self.method.lower()}")(data)

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

            self.dict_column[self.method][var] = {
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

            self.dict_column[self.method][var] = {
                "mean": mean,
                "std_dev": std_dev,
                "lb": lb,
                "ub": ub,
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_iforest_lof(self, data):

        for var in self.numerical_features:
            data_reshaped = data[var].values.reshape(-1, 1)
            labels = self.estimator.fit_predict(data_reshaped)
            outliers = np.where(labels == -1)[0]
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            self.dict_column[self.method][var] = {
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

            self.dict_column[self.method][var] = {
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_categorical(self, data):
        """
        Identify outliers in a categorical variable based on frequency.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data.

        Returns
        -------
        list
            List of outlier categories.
        """

        for var in self.categorical_features:
            value_counts = data[var].value_counts(normalize=True)
            series = value_counts[value_counts < 0.05]
            categories = list(series.index)
            n_outliers = len(data[data[var].isin(categories)]) if categories else 0
            p_outliers = (n_outliers / len(data)) * 100
            self.dict_column_cat[var] = {
                "n_outliers": n_outliers,
                "p_outliers": p_outliers,
                "categories": sorted(list(series.index)) if n_outliers > 0 else np.nan
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

    def get_results(self, method=None):
        """
        Provide a summary of outliers for all numerical variables.

        Returns
        -------
        pandas.DataFrame
            Summary of outlier detection for each numerical variable.
        """

        if method is not None:
            if method not in self.dict_column:
                raise ValueError(f"Method {method} has not been applied.")

        if self.method != "all":
            results = self.dict_column[self.method]
        else:
            if method is None:
                results = {}
                for method, variables in self.dict_column.items():
                    for var, metrics in variables.items():
                        if var not in results:
                            results[var] = {}
                        results[var][f"{method} n_outliers"] = metrics.get("n_outliers", np.nan)
                        results[var][f"{method} p_outliers"] = metrics.get("p_outliers", np.nan)
                return  pd.DataFrame.from_dict(results, orient="index")
            else:
                results = self.dict_column[method]

        return pd.DataFrame(results).T

    def get_results_cat_vars(self):
        """
        Provide a summary of outliers for all categorical variables.

        Returns
        -------
        pandas.DataFrame
            Summary of outlier detection for each catgorical variable.
        """

        results_cat = pd.DataFrame(self.dict_column_cat).T

        return results_cat

    def transform(self, data, method, metric="median"):
        """
        Impute outliers with the median or mean value for each variable based on the selected method.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data.

        method : str
            The method to use for identifying outliers. It must be one of the applied methods.

        metric : str or dict
            It could be median or mean, and it will apply for all variables. If it's a dictionary, it will apply
            the selected metric for each variable.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with outliers replaced by the median.
        """

        if method not in self.dict_column:
            raise ValueError(f"Method {method} has not been applied. "
                             f"Available methods: {list(self.dict_column.keys())}")

        if isinstance(metric, str):
            if metric not in ["median", "mean"]:
                raise ValueError("`metric` should be 'median' or 'mean'.")
        elif not isinstance(metric, dict):
            raise TypeError("`metric` should be string or dictionary.")

        def get_metric(data_transformed, metric, var):
            if isinstance(metric, str):
                if metric == "median":
                    _metric = data_transformed[var].median()
                else:
                    _metric = data_transformed[var].mean()
            elif isinstance(metric, dict):
                if metric[var] == "median":
                    _metric = data_transformed[var].median()
                elif metric[var] == "mean":
                    _metric = data_transformed[var].mean()
                else:
                    raise ValueError(f"`metric` is not valid for variable var, {var}")
            return _metric

        data_transformed = data.copy()

        for var, metrics in self.dict_column[method].items():
            if "lb" in metrics and "ub" in metrics:
                # For methods like IQR and std
                lb = metrics["lb"]
                ub = metrics["ub"]
                _metric = get_metric(data_transformed, metric, var)
                # Replace outliers with the median or mean
                data_transformed[var] = data_transformed[var].apply(lambda x: _metric if x < lb or x > ub else x)
            elif "n_outliers" in metrics:
                # For methods like IForest, LOF, GMM
                labels = self.estimator.fit_predict(data[var].values.reshape(-1, 1))
                outlier_indices = np.where(labels == -1)[0]
                _metric = get_metric(data_transformed, metric, var)
                # Replace outliers with the median or mean
                data_transformed.loc[outlier_indices, var] = _metric

        return data_transformed

    def combine_categories(self, data, var, rare_categories, new_category):
        """
        Combina categorías poco frecuentes en una categoría existente o nueva.

        Parameters
        ----------
        data : pandas.DataFrame
            Dataset original.
        var : str
            Nombre de la variable categórica.
        rare_categories : list
            Lista de categorías a combinar.
        new_category : str
            Nombre de la categoría combinada.

        Returns
        -------
        pandas.DataFrame
            DataFrame con las categorías combinadas.
        """
        data[var] = data[var].apply(lambda x: new_category if x in rare_categories else x)

        return data

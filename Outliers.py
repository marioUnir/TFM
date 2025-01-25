
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture


class OutliersDetection:
    """
    Clase para la detección de valores atípicos (outliers) en variables numéricas y categóricas.

    Esta clase implementa múltiples métodos para detectar valores atípicos en conjuntos de datos,
    incluyendo técnicas estadísticas y modelos basados en machine learning.

    :param method: Método para detectar valores atípicos. Puede ser uno de los siguientes:

        - IQR: Rango intercuartílico.
        - std: Desviación estándar.
        - IForest: Isolation Forest.
        - LOF: Local Outlier Factor.
        - GMM: Gaussian Mixture Model.
        - all: Aplica todos los métodos disponibles.
    :type method: str
    :param k: Valor multiplicador utilizado en los métodos estadísticos (IQR o std).

        - Si `method` es IQR, los límites se calculan como:

        .. math::

            \\text{lower limit} = Q_1 - k \\cdot (Q_3 - Q_1) \\\\
            \\text{upper limit} = Q_3 + k \\cdot (Q_3 - Q_1)

        - Si `method` es std, los límites se calculan como:

        .. math::

            \\text{lower limit} = \\mu - k \\cdot \\sigma \\\\
            \\text{upper limit} = \\mu + k \\cdot \\sigma
    :type k: float, optional (default=1.5)
    :param threshold: Umbral para identificar valores atípicos en variables categóricas, basado en la frecuencia relativa.
    :type threshold: float, optional (default=0.05)
    :param seed: Semilla para los métodos que requieren aleatoriedad.
    :type seed: int, optional (default=42)
    :param estimator: Estimador personalizado. Puede ser una instancia de `IsolationForest`, `LocalOutlierFactor` 
        o `GaussianMixture`.
    :type estimator: object, optional (default=None)
    :param contamination: Proporción de datos considerados como valores atípicos para los métodos basados en estimadores.
    :type contamination: float, optional (default=0.05)
    :param n_neighbors: Número de vecinos para el método Local Outlier Factor.
    :type n_neighbors: int, optional (default=20)
    :param n_components: Número de componentes en el modelo Gaussian Mixture.
    :type n_components: int, optional (default=2)

    :ivar numerical_features: Lista de nombres de variables numéricas en el conjunto de datos.
    :vartype numerical_features: list
    :ivar categorical_features: Lista de nombres de variables categóricas en el conjunto de datos.
    :vartype categorical_features: list
    :ivar dict_column: Diccionario con los resultados de la detección de outliers en variables numéricas.
    :vartype dict_column: dict
    :ivar dict_column_cat: Diccionario con los resultados de la detección de outliers en variables categóricas.
    :vartype dict_column_cat: dict
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
        # Inicialización de parámetros clave
        self.method = method
        self.k = k
        self.threshold = threshold
        self.seed = seed
        self.estimator = estimator
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.n_components = n_components

        # Inicializa atributos para almacenar resultados
        self.numerical_features = None
        self.categorical_features = None
        self.dict_column = {}
        self.dict_column_cat = {}

        # Verifica que los parámetros sean válidos
        self._check_parameters()

    def _check_parameters(self):
        """
        Verifica que los parámetros iniciales sean válidos.
        """
        if self.method not in ["IQR", "std", "IForest", "LOF", "GMM", "all"]:
            raise ValueError("`method` debe ser 'IQR', 'std', 'IForest', 'LOF', 'GMM' o 'all'.")

        if not isinstance(self.k, (float, int)) and self.method in ["IQR", "std"]:
            raise TypeError("`k` debe ser un número si se utiliza el método 'IQR' o 'std'.")

        if self.estimator is not None:
            # Comprueba que el estimador sea válido
            if isinstance(self.estimator, IsolationForest):
                self.method = "IForest"
            elif isinstance(self.estimator, LocalOutlierFactor):
                self.method = "LOF"
            elif isinstance(self.estimator, GaussianMixture):
                self.method = "GMM"
            else:
                raise ValueError("`estimator` debe ser una instancia de IsolationForest, LocalOutlierFactor o GaussianMixture.")

    def run(self, data):
        """
        Ejecuta el análisis de detección de valores atípicos.

        :param data: Conjunto de datos sobre el que se detectarán valores atípicos.
        :type data: pandas.DataFrame
            
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` debe ser un DataFrame de pandas.")

        # Identifica las variables numéricas y categóricas
        self.numerical_features, self.categorical_features = self._get_numerical_categorical_features(data)

        # Ejecuta la detección de valores atípicos
        self._detect_outliers(data)

    def _get_numerical_categorical_features(self, data):
        """
        Identifica variables numéricas y categóricas en el conjunto de datos.

        :param data: Conjunto de datos sobre el que se detectarán valores atípicos.
        :type data: pandas.DataFrame

        :return: Listas de variables numéricas y categóricas.
        :rtype: tuple
        """
        numerical_vars = data.select_dtypes(include=['number']).columns.tolist()
        categorical_vars = data.select_dtypes(exclude=['number']).columns.tolist()

        # Incluye variables numéricas que tienen menos de 10 valores únicos como categóricas
        categorical_vars += [col for col in numerical_vars if data[col].nunique() < 10]
        numerical_vars = [col for col in numerical_vars if col not in categorical_vars]

        return numerical_vars, categorical_vars

    def _initialize_estimator(self):
        """
        Inicializa el estimador para métodos que lo requieran.
        """
        if self.estimator is None:
            if self.method == "IForest":
                self.estimator = IsolationForest(contamination=self.contamination, random_state=self.seed)
            elif self.method == "LOF":
                self.estimator = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
            elif self.method == "GMM":
                self.estimator = GaussianMixture(n_components=self.n_components, random_state=self.seed)

    def _detect_outliers(self, data):
        """
        Detecta valores atípicos según el método seleccionado.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        print("** Variables Numéricas **")
        self._detect_outliers_numerical(data)  # Detección en variables numéricas

        print("** Variables Categóricas **")
        self._detect_outliers_categorical(data)  # Detección en variables categóricas

    def _detect_outliers_numerical(self, data):
        """
        Detecta valores atípicos en variables numéricas según el método.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        if self.method == "all":
            # Ejecuta todos los métodos y almacena resultados
            self.dict_column = {}
            all_methods = ["IQR", "std", "IForest", "LOF", "GMM"]
            for method in all_methods:
                print(f"Ejecutando método {method}...")
                self.method = method
                self._initialize_estimator()  # Inicializa el estimador si es necesario
                self.dict_column[self.method] = {}
                if method in ["IForest", "LOF", "GMM"]:
                    self._detect_outliers_iforest_lof(data) if method != "GMM" else self._detect_outliers_gmm(data)
                else:
                    getattr(self, f"_detect_outliers_{method.lower()}")(data)
            self.method = "all"  # Resetea el método a 'all'
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
        Detecta valores atípicos usando el método del rango intercuartílico (IQR).

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        for var in self.numerical_features:
            # Calcula el rango intercuartílico
            Q1 = data[var].quantile(0.25)
            Q3 = data[var].quantile(0.75)
            IQR = Q3 - Q1

            # Calcula los límites
            lb = Q1 - self.k * IQR
            ub = Q3 + self.k * IQR

            # Identifica los outliers
            outliers = data[(data[var] < lb) | (data[var] > ub)].index.tolist()
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            # Guarda resultados
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
        Detecta valores atípicos usando el método de desviación estándar.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        for var in self.numerical_features:
            # Calcula la media y la desviación estándar
            mean = data[var].mean()
            std_dev = data[var].std()

            # Calcula los límites
            lb = mean - self.k * std_dev
            ub = mean + self.k * std_dev

            # Identifica los outliers
            outliers = data[(data[var] < lb) | (data[var] > ub)].index.tolist()
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            # Guarda resultados en el diccionario
            self.dict_column[self.method][var] = {
                "mean": mean,
                "std_dev": std_dev,
                "lb": lb,
                "ub": ub,
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_iforest_lof(self, data):
        """
        Detecta valores atípicos utilizando Isolation Forest o Local Outlier Factor.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        for var in self.numerical_features:
            # Prepara los datos como un array 2D (necesario para los modelos)
            data_reshaped = data[var].values.reshape(-1, 1)

            # Genera las predicciones (-1 indica outliers)
            labels = self.estimator.fit_predict(data_reshaped)
            outliers = np.where(labels == -1)[0]
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            # Guarda los resultados
            self.dict_column[self.method][var] = {
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_gmm(self, data):
        """
        Detecta valores atípicos utilizando Gaussian Mixture Model.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        for var in self.numerical_features:
            # Prepara los datos como un array 2D (necesario para los modelos)
            data_reshaped = data[var].values.reshape(-1, 1)

            # Ajusta el modelo y calcula las puntuaciones
            self.estimator.fit(data_reshaped)
            scores = self.estimator.score_samples(data_reshaped)

            # Calcula un umbral basado en el percentil inferior (5% por defecto)
            threshold = np.percentile(scores, 5)
            outliers = np.where(scores < threshold)[0]
            outliers_count = len(outliers)
            outliers_perc = (outliers_count / len(data)) * 100

            # Guarda los resultados
            self.dict_column[self.method][var] = {
                "n_outliers": outliers_count,
                "p_outliers": outliers_perc,
            }

    def _detect_outliers_categorical(self, data):
        """
        Identifica valores atípicos en variables categóricas basados en su frecuencia relativa.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos en el que se detectarán valores atípicos.
        """
        for var in self.categorical_features:
            # Calcula la frecuencia relativa de cada categoría
            value_counts = data[var].value_counts(normalize=True)

            # Filtra categorías con frecuencias menores al umbral
            series = value_counts[value_counts < self.threshold]
            categories = list(series.index)

            # Cuenta los valores atípicos en estas categorías
            n_outliers = len(data[data[var].isin(categories)]) if categories else 0
            p_outliers = (n_outliers / len(data)) * 100

            # Guarda resultados en el diccionario
            self.dict_column_cat[var] = {
                "n_outliers": n_outliers,
                "p_outliers": p_outliers,
                "categories": sorted(list(series.index)) if n_outliers > 0 else np.nan
            }

    def get_outliers(self, variable):
        """
        Obtiene la lista de índices de valores atípicos para una variable específica.

        :param variable: Nombre de la variable de la cual se quieren obtener los índices de valores atípicos.
        :type variable: str Parameters

        :return: Lista de índices de los valores atípicos.
        :rtype: list

        :raises ValueError: Si la variable no ha sido analizada previamente.                   
        """
        # Verifica si la variable fue analizada
        if variable not in self.dict_column:
            raise ValueError(f"La variable {variable} no ha sido analizada.")

        # Devuelve los índices de los outliers detectados
        return self.dict_column[variable].get("outliers", [])

    def get_results(self, method=None):
        """
        Devuelve un resumen de los valores atípicos detectados para variables numéricas.

        :param method: Método específico para el cual se quieren los resultados.
            Si no se especifica, se devolverán resultados de todos los métodos.
        :type method: str (default=None)

        :return: Resumen de los valores atípicos detectados por variable y método.
        :rtype: pd.DataFrame           
        """
        if method is not None:
            if method not in self.dict_column:
                raise ValueError(f"El método {method} no ha sido aplicado.")

        if self.method != "all":
            # Retorna resultados para el método actual
            results = self.dict_column[self.method]
        else:
            # Combina resultados de todos los métodos
            if method is None:
                results = {}
                for method, variables in self.dict_column.items():
                    for var, metrics in variables.items():
                        if var not in results:
                            results[var] = {}
                        results[var][f"{method} n_outliers"] = metrics.get("n_outliers", np.nan)
                        results[var][f"{method} p_outliers"] = metrics.get("p_outliers", np.nan)
                return pd.DataFrame.from_dict(results, orient="index")
            else:
                results = self.dict_column[method]

        return pd.DataFrame(results).T

    def get_results_cat_vars(self):
        """
        Devuelve un resumen de los valores atípicos detectados en variables categóricas.

        :return: Resumen de los valores atípicos detectados para cada variable categórica.
        :rtype: pd.DataFrame      
        """
        # Convierte el diccionario de resultados a un DataFrame
        results_cat = pd.DataFrame(self.dict_column_cat).T
        return results_cat

    def transform(self, data, method, metric="median"):
        """
        Sustituye los valores atípicos detectados con la mediana o la media.

        :param data: Conjunto de datos original.
        :type data: pandas.DataFrame
        :param method: Método utilizado para identificar los valores atípicos.
        :type method: str
        :param metric: Puede ser "median" o "mean" para aplicar el mismo valor a todas las variables, 
            o un diccionario para aplicar una métrica diferente a cada variable. Por defecto, "median".
        :type metric: str | dict, optional

        :return: Nuevo DataFrame con los valores atípicos sustituidos.
        :rtype: pandas.DataFrame
        """

        if method not in self.dict_column:
            raise ValueError(f"El método {method} no ha sido aplicado. "
                             f"Métodos disponibles: {list(self.dict_column.keys())}")

        if isinstance(metric, str):
            if metric not in ["median", "mean"]:
                raise ValueError("`metric` debe ser 'median' o 'mean'.")
        elif not isinstance(metric, dict):
            raise TypeError("`metric` debe ser un string o un diccionario.")

        def get_metric(data_transformed, metric, var):
            """
            Calcula la mediana o media para la variable especificada.

            Parameters
            ----------
            data_transformed : pandas.DataFrame
                DataFrame transformado.

            metric : str o dict
                Métrica a utilizar.

            var : str
                Nombre de la variable.

            Returns
            -------
            float
                Valor de la métrica.
            """
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
                    raise ValueError(f"`metric` no es válido para la variable {var}")
            return _metric

        # Copia el DataFrame original
        data_transformed = data.copy()

        # Reemplaza los valores atípicos
        for var, metrics in self.dict_column[method].items():
            if "lb" in metrics and "ub" in metrics:
                # Para métodos como IQR y std
                lb = metrics["lb"]
                ub = metrics["ub"]
                _metric = get_metric(data_transformed, metric, var)
                data_transformed[var] = data_transformed[var].apply(lambda x: _metric if x < lb or x > ub else x)
            elif "n_outliers" in metrics:
                # Para métodos como IForest, LOF, GMM
                labels = self.estimator.fit_predict(data[var].values.reshape(-1, 1))
                outlier_indices = np.where(labels == -1)[0]
                _metric = get_metric(data_transformed, metric, var)
                data_transformed.loc[outlier_indices, var] = _metric

        return data_transformed

    def combine_categories(self, data, var, rare_categories, new_category):
        """
        Combina categorías poco frecuentes en una categoría existente o nueva.

        :param data: Conjunto de datos original.
        :type data: pandas.DataFrame
        :param var: Nombre de la variable categórica.
        :type var: str
        :param rare_categories: Lista de categorías que se combinarán.
        :type rare_categories: list
        :param new_category: Nombre de la nueva categoría que representará las categorías combinadas.
        :type new_category: str

        :return: DataFrame modificado con las categorías combinadas.
        :rtype: pandas.DataFrame
        """

        # Reemplaza las categorías poco frecuentes por la nueva categoría
        data[var] = data[var].apply(lambda x: new_category if x in rare_categories else x)
        return data


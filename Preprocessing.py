import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

class Preprocessing:

    """
    Clase para preprocesar datos numéricos y categóricos.

    Esta clase implementa estrategias para imputar valores faltantes, codificar variables categóricas 
    y escalar variables numéricas.

    :param target: Nombre de la variable objetivo que no será transformada.
    :type target: str
    :param missing_strategy_num: Estrategia de imputación para características numéricas. 
        Puede ser "mean", "median", "most_frequent" o "constant".
    :type missing_strategy_num: str (default="median")
    :param missing_strategy_cat: Estrategia de imputación para características categóricas. 
        Puede ser "most_frequent" o "constant".
    :type missing_strategy_cat: str (default="most_frequent")
    :param fill_value_num: Valor constante utilizado para imputar valores numéricos si 
        `missing_strategy_num` es "constant".
    :type fill_value_num: float | int, optional (default=None)
    :param fill_value_cat: Valor constante utilizado para imputar valores categóricos si 
        `missing_strategy_cat` es "constant".
    :type fill_value_cat: str, optional (default=None)
    :param categorical_encoding: Método de codificación para características categóricas. 
        Puede ser "onehot", "label" o "dummies".
    :type categorical_encoding: str (default="onehot")
    :param scaler_method: Método para escalar variables numéricas. Puede ser "minmax", 
        "standard" o "robust".
    :type scaler_method: str (default="minmax")

    :ivar num_imputer: Instancia de `SimpleImputer` utilizada para imputar valores numéricos.
    :vartype num_imputer: sklearn.impute.SimpleImputer
    :ivar cat_imputer: Instancia de `SimpleImputer` utilizada para imputar valores categóricos.
    :vartype cat_imputer: sklearn.impute.SimpleImputer
    :ivar encoder: Codificador utilizado para variables categóricas. 
        Puede ser una instancia de `OneHotEncoder` o un diccionario de `LabelEncoder`.
    :vartype encoder: sklearn.preprocessing.OneHotEncoder | dict
    :ivar scaler: Escalador utilizado para normalizar variables numéricas.
    :vartype scaler: sklearn.preprocessing.MinMaxScaler | sklearn.preprocessing.StandardScaler | sklearn.preprocessing.RobustScaler
    """

    def __init__(
            self, 
            target,
            missing_strategy_num="median", 
            missing_strategy_cat="most_frequent", 
            fill_value_num=None,
            fill_value_cat=None,
            categorical_encoding="onehot",
            scaler_method="minmax"
        ):

        self.target = target
        self.missing_strategy_num = missing_strategy_num
        self.missing_strategy_cat = missing_strategy_cat
        self.fill_value_num = fill_value_num
        self.fill_value_cat = fill_value_cat
        
        self.categorical_encoding = categorical_encoding
        self.scaler_method = scaler_method
        
        # Inicializar estimador
        self.num_imputer = SimpleImputer(strategy=self.missing_strategy_num, fill_value=self.fill_value_num)
        self.cat_imputer = SimpleImputer(strategy=self.missing_strategy_cat, fill_value=self.fill_value_cat)
        
        # Inicializar atributos
        self.encoder = None

        self._check_parameters()

    def _check_parameters(self):

        """
        Verifica que los parámetros iniciales de la clase sean válidos.

        Este método valida los valores proporcionados para las estrategias de imputación, 
        codificación categórica y escalado numérico. Si algún valor no es válido, 
        se lanzará un error.

        :raises ValueError: Si alguno de los parámetros no tiene un valor válido.
        """

        # missing_strategy_num 
        possible = ["mean", "median", "most_frequent", "constant"]
        if self.missing_strategy_num not in possible:
            raise ValueError(f"`missing_strategy_num` debe ser algún valor de {possible}")
        if self.missing_strategy_num == "constant" and self.fill_value_num is None:
            raise ValueError("`fill_value_num` debe estar informado cuando `missing_strategy_num` es constant.")
        
        # missing_strategy_cat
        possible = ["most_frequent", "constant"]
        if self.missing_strategy_cat not in possible:
            raise ValueError(f"`missing_strategy_cat` debe ser algún valor de {possible}")
        if self.missing_strategy_num == "constant" and self.fill_value_cat is None:
            raise ValueError("`fill_value_cat` debe estar informado cuando `missing_strategy_cat` es constant.")
        
        # categorical_encoding
        possible = ["onehot", "label", "dummies"]
        if self.categorical_encoding not in possible:
            raise ValueError(f"`categorical_encoding` debe ser algún valor de {possible}") 
        
        # scaler_method
        possible = ["minmax", "standard", "robust"]
        if self.scaler_method not in possible:
            raise ValueError(f"`scaler_method` debe ser algún valor de {possible}") 

    def _get_variable_types(self, data):
        """
        Identifica variables numéricas y categóricas
        """
        
        self.column_names = list(data.columns)

        numerical_vars = data.select_dtypes(
            include=['number']).columns.tolist()
        categorical_vars = data.select_dtypes(
            exclude=['number']).columns.tolist()

        # Agrega variables explícitamente si algunos valores numéricos son categóricos 
        # (por ejemplo, Género, Matrimonio).
        categorical_vars += [
            col for col in numerical_vars if data[col].nunique() < 10
        ]
        numerical_vars = [
            col for col in numerical_vars if col not in categorical_vars
        ]

        self.categorical_features = [
            x for x in categorical_vars if x != self.target
        ]
        self.numerical_features = [
            x for x in numerical_vars if  x != self.target
        ]

    def fit_transform(self, data, numerical_columns=None, categorical_columns=None):
        """
        Ajusta y transforma los datos.

        Este método realiza imputación de valores faltantes, codificación de variables categóricas
        y escalado de variables numéricas en el conjunto de datos proporcionado.

        :param data: Conjunto de datos de entrada.
        :type data: pandas.DataFrame
        :param numerical_columns: Lista de nombres de columnas numéricas. Si es `None`, 
            las columnas numéricas serán detectadas automáticamente.
        :type numerical_columns: list[str], optional (default=None)
        :param categorical_columns: Lista de nombres de columnas categóricas. Si es `None`, 
            las columnas categóricas serán detectadas automáticamente.
        :type categorical_columns: list[str], optional (default=None)

        :return: DataFrame transformado, con valores imputados, variables categóricas codificadas 
            y variables numéricas escaladas.
        :rtype: pandas.DataFrame
        """

        self._get_variable_types(data)

        if numerical_columns is None:
            numerical_columns = self.numerical_features

        if categorical_columns is None:
            categorical_columns = self.categorical_features

        # Paso 1 : Imputando missings
        data = self._impute_missings(
            data=data, 
            numerical_columns=numerical_columns, 
            categorical_columns=categorical_columns
        )

        # Paso 2 : Codifica variables categóricas
        data = self._encoding_cat_vars(
            data=data,
            categorical_columns=categorical_columns
        )

        # Paso 3 : Escalar variables numéricas
        data = self._scale_num_vars(
            data=data,
            numerical_columns=numerical_columns
        )


        return data
    
    def _impute_missings(self, data, numerical_columns, categorical_columns):
        """
        Imputa valores faltantes en columnas numéricas y categóricas.

        Este método utiliza los imputadores definidos en la clase (`SimpleImputer`)
        para rellenar valores faltantes en las columnas numéricas y categóricas.

        :param data: Conjunto de datos de entrada.
        :type data: pandas.DataFrame
        :param numerical_columns: Lista de nombres de columnas numéricas a imputar.
        :type numerical_columns: list[str]
        :param categorical_columns: Lista de nombres de columnas categóricas a imputar.
        :type categorical_columns: list[str]

        :return: DataFrame con los valores faltantes imputados.
        :rtype: pandas.DataFrame
        """

        print("Imputando valores ...")

        # Imputar valores faltantes en columnas numéricas
        data[numerical_columns] = self.num_imputer.fit_transform(data[numerical_columns])

        # Imputar valores faltantes en columnas categóricas
        data[categorical_columns] = self.cat_imputer.fit_transform(data[categorical_columns])

        return data
    
    def _encoding_cat_vars(self, data, categorical_columns):
        """
        Codifica variables categóricas utilizando el método especificado.

        Este método transforma las columnas categóricas según el método de codificación definido en 
        `categorical_encoding`, que puede ser "onehot", "label", o "dummies".

        :param data: Conjunto de datos de entrada.
        :type data: pandas.DataFrame
        :param categorical_columns: Lista de nombres de columnas categóricas a codificar.
        :type categorical_columns: list[str]

        :return: DataFrame con las variables categóricas codificadas.
        :rtype: pandas.DataFrame
        """

        print("Codificando variables categóricas ...")

        # Codificar columnas categóricas
        if self.categorical_encoding == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, drop=None)
            encoded = self.encoder.fit_transform(data[categorical_columns])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(categorical_columns),
                index=data.index,
            )
            data = pd.concat([data.drop(columns=categorical_columns), encoded_df], axis=1)

        elif self.categorical_encoding == "label":
            self.encoder = {
                col: LabelEncoder().fit(data[col]) for col in categorical_columns
            }
            for col, le in self.encoder.items():
                data[col] = le.transform(data[col])
        
        elif self.categorical_encoding == "dummies":
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

        return data
    
    def _scale_num_vars(self, data, numerical_columns):
        """
        Escala variables numéricas utilizando el método especificado.

        Este método transforma las columnas numéricas según el método de escalado definido en 
        `scaler_method`, que puede ser "minmax", "standard", o "robust".

        :param data: Conjunto de datos de entrada.
        :type data: pandas.DataFrame
        :param numerical_columns: Lista de nombres de columnas numéricas a escalar.
        :type numerical_columns: list[str]

        :return: DataFrame con las variables numéricas escaladas.
        :rtype: pandas.DataFrame
        """

        print("Escalando variables numéricas ...")
        
        if self.scaler_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_method == "robust":
            self.scaler = RobustScaler()
        
        data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])

        return data

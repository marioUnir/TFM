import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler

class Preprocessing:
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
        """
        Inicializa la clase Preprocessing.

        Parámetros
        ----------
        target : str
            Variable objetivo que no se verá afectada
        missing_strategy_num : str (default="median")
            Estrategia de imputación para características numéricas ("mean", "median", "most_frequent", "constant").
        missing_strategy_cat : str (default="most-frequent")
            Estrategia de imputación para características numéricas ("most_frequent", "constant").
        fill_value_num : numerical (default=None)
            Cuando ``missing_strategy_num`` es constant, será usado para reemplazar los missings por este valor
        fill_value_cat : str (default=None)
            Cuando ``missing_strategy_cat`` es constant, será usado para reemplazar los missings por este valor
        categorical_encoding : str (default="onehot")
            Método de codificación para características categóricas ("onehot", "label" o "dummies").
        scaler_method : str (default="minmax")
            Método para escalar variables numéricas ("minmax", "standard", "robust")
        """
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

        Parámetros
        ----------

        data :  pd.DataFrame 
            El DataFrame de entrada.
        numerical_columns : list (default=None)
            Lista de nombres de columnas numéricas. Si es None, se detectan automáticamente.
        categorical_columns : list (default=None)
            Lista de nombres de columnas categóricas. Si es None, se detectan automáticamente.

        Retorna:
        pd.DataFrame: DataFrame transformado.
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
        
        print("Imputando valores ...")

        # Imputar valores faltantes en columnas numéricas
        data[numerical_columns] = self.num_imputer.fit_transform(data[numerical_columns])

        # Imputar valores faltantes en columnas categóricas
        data[categorical_columns] = self.cat_imputer.fit_transform(data[categorical_columns])

        return data
    
    def _encoding_cat_vars(self, data, categorical_columns):
        
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

        print("Escalando variables numéricas ...")
        
        if self.scaler_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_method == "robust":
            self.scaler = RobustScaler()
        
        data[numerical_columns] = self.scaler.fit_transform(data[numerical_columns])

        return data

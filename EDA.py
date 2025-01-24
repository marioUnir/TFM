import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAProcessor:

    """
    Clase para procesar tareas de análisis exploratorio de datos (EDA).

    Attributes
    ----------
        dict_column : dict
            Diccionario para almacenar metadatos de las columnas.
        n_samples : int
            Número de filas en el conjunto de datos.
        n_columns : int
            Número de columnas en el conjunto de datos.
        column_names : list[str]
            Lista con los nombres de las columnas.
        categorical_features : list[str]
            Lista de columnas categóricas.
        numerical_features : list[str]
            Lista de columnas numéricas.

    Parameters
    ----------
        target : str: 
            Nombre de la columna objetivo.
        ordinal_columns : list[str] 
            Lista de columnas ordinales.
        exclude_columns : list[str] | None (default=None)
            Columnas a excluir del análisis.
        verbose : bool (default=True)
            Indica si se imprimen mensajes durante el procesamiento.
    """

    def __init__(
            self,
            target,
            ordinal_columns,
            exclude_columns=None,
            verbose=True
    ):

        self.target = target
        self.ordinal_columns = ordinal_columns
        self.exclude_columns = exclude_columns
        self.verbose = verbose

        self.dict_column = {}

        self.n_samples = None
        self.n_columns = None
        self.column_names = None

        self.categorical_features = None
        self.numerical_features = None

    def _initialize_dict_column(self):
        """
        Inicializa el diccionario para almacenar metadatos de cada columna.

        Cada columna tendrá información como tipo de dato, cantidad y porcentaje de valores
        faltantes, número de valores únicos y si tiene duplicados.
        """

        dict_column_info = {
            "dtype": None,
            "n_missing": 0,
            "p_missing": 0.0,
            "n_unique": 0,
            "is_duplicate": False
        }

        # initialize dict of columns
        for column in self.column_names:
            self.dict_column[column] = dict_column_info.copy()

    def run(self, data):
        """
        Ejecuta el análisis exploratorio de datos.

        El EDA realiza tres pasos: 

        1. Identifica el tipo de variable (numérica, categórica, etc.).
        2. Calcula valores faltantes y duplicados.
        3. Calcula la matriz de correlación.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos a analizar.

        Raises
        ------
            TypeError: Si el argumento `data` no es un DataFrame de pandas.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` debe ser un DataFrame de Pandas.")

        self.n_samples = len(data)
        self.n_columns = len(data.columns)
        self.column_names = list(data.columns)

        self._initialize_dict_column()

        if self.verbose:
            print("Empezando EDA ...\n")

        # Paso 1: Identifica variables numéricas y categóricas
        self._get_variable_types(data)

        # Paso 2: Encuentra missings y duplicados
        self._calculate_missing_and_duplicates(data)

        # Paso 3: Correlaciones
        self._calculate_correlations(data)

        if self.verbose:
            print("EDA completado ...\n")

    def _get_variable_types(self, data):
        """
        Identifica variables numéricas y categóricas

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos a analizar.
        """

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

        for col in self.column_names:
            if col == self.target:
                self.dict_column[col]["dtype"] = "target"
            elif col in numerical_vars:
                if col in self.ordinal_columns:
                    self.dict_column[col]["dtype"] = "ordinal"
                else:
                    self.dict_column[col]["dtype"] = "numerical"
            elif col in categorical_vars:
                if col in self.ordinal_columns:
                    self.dict_column[col]["dtype"] = "ordinal"
                else:
                    self.dict_column[col]["dtype"] = "categorical"

        self.categorical_features = [
            x for x in categorical_vars if x not in self.ordinal_columns
            and x != self.target and x not in self.exclude_columns
        ]
        self.numerical_features = [
            x for x in numerical_vars if x not in self.ordinal_columns
            and x != self.target and x not in self.exclude_columns
        ]

    def _calculate_missing_and_duplicates(self, data):
        """
        Calcula los valores faltantes y duplicados para cada columna y 
        se almacenan en dict_column.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos a analizar.
        """
        for col in self.column_names:
            missing_count = data[col].isnull().sum()
            self.dict_column[col]["n_missing"] = missing_count
            self.dict_column[col]["p_missing"] = (missing_count /
                                                  self.n_samples) * 100
            self.dict_column[col]["n_unique"] = data[col].nunique()

        # Comprueba filas duplicada y actualiza ``dict_column``
        is_duplicate = data.duplicated().any()
        for col in self.column_names:
            self.dict_column[col]["is_duplicate"] = is_duplicate

    def _calculate_correlations(self, data):
        """
        Calcula la matriz de correlación para las variables numéricas.

        Parameters
        ----------
        data : pandas.DataFrame
            Conjunto de datos a analizar.
        """

        # Calcula la correlación para las variables numéricas
        corr = data[self.numerical_features + [self.target]].corr()
        self.df_corr = corr

    def distribution_variable(self, data, column, n_bins=None, bins=None):
        """
        Calcula la distribución de una variable dividiéndola en intervalos.

        Este método permite dividir los valores de una columna en intervalos (bins) 
        y calcular la frecuencia de valores dentro de cada intervalo.

        Parameters
        ----------
            data: pd.DataFrame 
                Conjunto de datos que contiene la columna a analizar.
            column: str
                Nombre de la columna para la cual calcular la distribución.
            n_bins:  int (default=None)
                Número de intervalos (bins) en los que dividir la columna.
                Si se proporciona, los intervalos se calcularán automáticamente usando valores equiespaciados.
            bins : list[float] | list[int] (default=None): 
                Lista de límites de los intervalos definidos manualmente. Si se proporciona, `n_bins` será ignorado.

        Returns
        -------
            pd.Series: Serie con la frecuencia de valores en cada intervalo, ordenada por los intervalos.

        Raises
        ------
            ValueError: Si el nombre de la columna no está en el conjunto de datos.
        """

        if column not in data.columns:
            raise ValueError("`column` debe ser un nombre válido de columna.")

        if n_bins is not None:
            min_value = data[column].min()
            max_value = data[column].max()
            bins = list(
                np.linspace(min_value, max_value + 1, n_bins + 1).astype(int))

        return pd.cut(data[column], bins=bins,
                      right=False).value_counts().sort_index()

    def plot_numerical_variables(self, data, num_vars=None, figsize=None):
        """
        Genera histogramas para las variables numéricas seleccionadas.

        Este método crea histogramas para analizar la distribución de las variables numéricas
        en el conjunto de datos. Si se seleccionan múltiples variables, se generarán subgráficos.

        Parameters
        ----------
            data: pd.DataFrame
                Conjunto de datos que contiene las variables a graficar.
            num_vars: list[str] (default=None)
                Lista de nombres de columnas numéricas a graficar. 
                Si no se proporciona, se utilizarán todas las variables numéricas identificadas.
            figsize: tuple (default=None)
                Dimensiones de la figura (ancho, alto). Si no se proporciona, se establecerán valores predeterminados.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera un gráfico directamente.

        Raises
        ------
            ValueError: Si alguna de las variables proporcionadas en `num_vars` no se encuentra en las variables numéricas identificadas.
        """

        if num_vars is None:
            num_vars = self.numerical_features
        else:
            bad_input = []
            for x in num_vars:
                if x not in self.numerical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("Las siguientes columnas no se consideran numéricas: "
                                 f"{bad_input}")

        num_graphs = len(num_vars)

        # Configurar el estilo de Seaborn
        sns.set(style="whitegrid")

        # Caso de un solo gráfico
        if num_graphs == 1:

            if figsize is None:
                figsize = (8, 5)
            
            plt.figure(figsize=figsize)
            sns.histplot(data[num_vars[0]], bins=30, kde=True, color='skyblue')
            plt.title(f'Distribución de {num_vars[0]}', fontsize=14, weight='bold')
            plt.xlabel(f'{num_vars[0]}')
            plt.ylabel('Frecuencia')

        else:

            if figsize is None:
                figsize = (12, num_graphs * 2.5)

            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=figsize)

            for i, var in enumerate(num_vars):
                # Histograma
                sns.histplot(data[var],
                             bins=30,
                             kde=True,
                             ax=axes[i],
                             color='skyblue')
                # Añadir título
                axes[i].set_title(f'Distribución de {var}',
                                  fontsize=14,
                                  weight='bold')
                # Añadir título al eje X
                axes[i].set_xlabel(f'{var}')
                # Añadir título al eje y
                axes[i].set_ylabel('Frequencia')

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

        plt.tight_layout()
        plt.show()

    def plot_categorical_variables(self, data, cat_vars=None):
        """
        Genera gráficos de barras para las variables categóricas seleccionadas.

        Este método crea gráficos de barras para analizar la distribución de las variables categóricas 
        en el conjunto de datos. Si se seleccionan múltiples variables, se generarán subgráficos.

        Parameters
        ----------
            data: pd.DataFrame
                Conjunto de datos que contiene las variables categóricas a graficar.
            cat_vars: list[str] (default=None)
                Lista de nombres de columnas categóricas a graficar. 
                Si no se proporciona, se utilizarán todas las variables categóricas identificadas.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera gráficos directamente.

        Raises
        ------
            ValueError: Si alguna de las variables proporcionadas en `cat_vars` no se encuentra en las variables categóricas identificadas.
        """

        if cat_vars is None:
            cat_vars = self.categorical_features
        else:
            bad_input = []
            for x in cat_vars:
                if x not in self.categorical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("Las siguientes columnas no se consideran categóricas: "
                                 f"{bad_input}")

        num_graphs = len(cat_vars)

        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.countplot(data, x=cat_vars[0], color='skyblue')
            plt.title(f'Distribución de {cat_vars[0]}', fontsize=14, weight='bold')
            plt.xlabel(f'{cat_vars[0]}')
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=90)
        else:
            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=(12, num_graphs * 2.5))

            for i, var in enumerate(cat_vars):
                sns.countplot(data, x=var, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Distribución de {var}',
                                  fontsize=14,
                                  weight='bold')
                axes[i].set_xlabel(f'{var}')
                axes[i].set_ylabel('Frecuencia')

                # Rotar las etiquetas del eje x
                axes[i].tick_params(axis='x', rotation=90)

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

        plt.tight_layout()
        plt.show()

    def plot_target(self, data):
        """
        Genera un gráfico de barras para analizar la distribución del target.

        Este método crea un gráfico de barras que muestra la frecuencia de cada categoría 
        en la columna objetivo (`target`) del conjunto de datos.

        Parameters
        ----------
            data: pd.DataFrame
                Conjunto de datos que contiene la columna objetivo.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera un gráfico directamente.
        """

        plt.figure(figsize=(10, 6))
        sns.countplot(data=data,
                      x=self.target,
                      hue=self.target,
                      palette="pastel")
        plt.title('Distribución del target: Default')
        plt.xlabel('Default')
        plt.ylabel('Frecuencia')

    def plot_correlations(self):
        """
        Genera un mapa de calor para visualizar las correlaciones entre variables numéricas.

        Este método utiliza la matriz de correlación calculada previamente (`self.df_corr`)
        para crear un mapa de calor, lo que facilita el análisis de relaciones entre variables numéricas.

        Parameters
        ----------
            None: Este método no requiere parámetros adicionales.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera un gráfico directamente.
        """

        plt.figure(figsize=(14, 10))

        sns.set(style="whitegrid")
        heatmap = sns.heatmap(self.df_corr,
                              annot=True,
                              cmap='coolwarm',
                              linewidths=0.5,
                              fmt=".1f",  # Usar un solo decimal para hacer los números más compactos
                              annot_kws={"size": 8})  # Ajusta el tamaño de la fuente de las anotaciones

        plt.title('Mapa de Calor de la Matriz de Correlación', fontsize=16, weight='bold')

        # Alineación de las etiquetas del eje x y y
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_numerical_vs_target(self, data, num_vars=None):
        """
        Genera gráficos de caja (boxplots) para analizar la relación entre variables numéricas y el target.

        Este método crea un gráfico de caja para cada variable numérica seleccionada, 
        mostrando la distribución de sus valores con respecto a las categorías de la columna objetivo (`target`).

        Parameters
        ----------
            data: pd.DataFrame
                Conjunto de datos que contiene las variables a analizar.
            num_vars: list[str] (default=None)
                Lista de nombres de columnas numéricas a analizar. 
                Si no se proporciona, se utilizarán todas las variables numéricas identificadas.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera gráficos directamente.

        Raises
        ------
            ValueError: Si alguna de las variables proporcionadas en `num_vars` no se encuentra en las variables numéricas identificadas.
        """

        if num_vars is None:
            num_vars = self.numerical_features
        else:
            bad_input = []
            for x in num_vars:
                if x not in self.numerical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("Las siguientes columnas no se consideran numéricas: "
                                 f"{bad_input}")

        num_graphs = len(num_vars)

        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=data, x=self.target, y=num_vars[0], color="skyblue", fill=False)
            plt.title(f'{num_vars[0]} vs Default')
            plt.xlabel('Default')
            plt.ylabel(num_vars[0])

        else:
            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=(12, num_graphs * 2.5))

            for i, var in enumerate(num_vars):
                sns.boxplot(data=data,
                            x=self.target,
                            y=var,
                            ax=axes[i],
                            color="skyblue",
                            fill=False)
                axes[i].set_title(f'{var.replace("_", " ")} vs Default')
                axes[i].set_xlabel('Default')
                axes[i].set_ylabel(var.replace("_", " "))
                axes[i].tick_params(axis='x')

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

        plt.tight_layout()
        plt.show()

    def plot_histogram_vs_target(self, data,cat_vars=None):
        """
        Genera gráficos de barras agrupados para analizar la relación entre variables categóricas y el target.

        Este método crea gráficos de barras para cada variable categórica seleccionada, mostrando
        la frecuencia de sus valores desglosados por las categorías de la columna objetivo (`target`).

        Parameters
        ----------
            data: pd.DataFrame
                Conjunto de datos que contiene las variables categóricas a analizar.
            cat_vars: list[str] (default=None)
                Lista de nombres de columnas categóricas a analizar. 
                Si no se proporciona, se utilizarán todas las variables categóricas identificadas.

        Returns
        -------
            None: Este método no retorna ningún valor. Genera gráficos directamente.

        Raises
        ------
            ValueError: Si alguna de las variables proporcionadas en `cat_vars` no se encuentra en las variables categóricas identificadas.
        """

        if cat_vars is None:
            cat_vars = self.categorical_features
        else:
            bad_input = []
            for x in cat_vars:
                if x not in self.categorical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("Las siguientes columnas no se consideran categóricas: "
                                 f"{bad_input}")

        num_graphs = len(cat_vars)

        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=cat_vars[0], hue=self.target, data=data, palette='Blues_d',
                          order=data[cat_vars[0]].value_counts().index)
            plt.title(f'Distribución de {cat_vars[0]} por Default')
            plt.xlabel(cat_vars[0])
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=90)
            for p in plt.gca().patches:
                height = p.get_height()
                if pd.notna(height):
                    plt.gca().annotate(f'{int(height)}',
                                       (p.get_x() + p.get_width() / 2., height),
                                       ha='center', va='center', xytext=(0, -6),
                                       textcoords='offset points', color='white', fontsize=9)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles, labels, title='Default')
        else:
            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=(12, num_graphs * 2.5))

            # Crear un gráfico para cada variable categórica
            for i, attribute in enumerate(cat_vars):
                sns.countplot(x=attribute,
                              hue=self.target,
                              data=data,
                              palette='Blues_d',
                              order=data[attribute].value_counts().index,
                              ax=axes[i])
                axes[i].set_title(f'Distribución de {attribute} por Default')
                axes[i].set_xlabel(attribute)
                axes[i].set_ylabel('Frecuencia')
                axes[i].tick_params(axis='x', rotation=90)
                # Eliminar la leyenda de cada subgráfico
                axes[i].get_legend().remove()
                # Añadir etiquetas de conteo
                for p in axes[i].patches:
                    height = p.get_height()
                    if pd.notna(height):  # Verificar si la barra no está vacía
                        axes[i].annotate(f'{int(height)}',
                                         (p.get_x() + p.get_width() / 2., height),
                                         ha='center',
                                         va='center',
                                         xytext=(0, -6),
                                         textcoords='offset points',
                                         color='white',
                                         fontsize=9)

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

            # Crear una leyenda única para toda la figura
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles,
                       labels,
                       title='Default',
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1.05),
                       ncol=4)

        # Ajustar el layout y mostrar el gráfico
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def get_results(self):
        """
        Obtén el resumen del análisis exploratorio de datos (EDA).

        Este método devuelve un DataFrame que contiene metadatos de cada columna en el conjunto de datos, incluyendo información como:

        - Tipo de dato
        - Número de valores faltantes
        - Porcentaje de valores faltantes
        - Número de valores únicos
        - Si la columna tiene filas duplicadas

        Returns
        -------
        pandas.DataFrame
            Un DataFrame donde cada fila corresponde a una columna en el conjunto de datos, 
            y los metadatos de cada columna se almacenan en la fila respectiva.
        """

        return pd.DataFrame(self.dict_column).T

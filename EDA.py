import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDAProcessor:

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
        Initialize the dictionary to store metadata for each column.
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
        Run preprocessing.

        Preprocessing performs three steps: categorizes each column data, and
        finds duplicates and nulls.

        Parameters
        ----------
        data : pandas.DataFrame
            Raw dataset.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe.")

        self.n_samples = len(data)
        self.n_columns = len(data.columns)
        self.column_names = list(data.columns)

        self._initialize_dict_column()

        if self.verbose:
            print("Starting EDA...\n")

        # Step 1: Identify numerical and categorical variables
        self._get_variable_types(data)

        # Step 2: Calculate missing values and duplicates
        self._calculate_missing_and_duplicates(data)

        # Step 3: Correlations
        self._calculate_correlations(data)

        if self.verbose:
            print("EDA completed ...\n")

    def _get_variable_types(self, data):
        """
        Identify numerical and categorical variables
        """

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
        Calculate missing values and duplicates for each column and store in dict_column.
        """
        for col in self.column_names:
            missing_count = data[col].isnull().sum()
            self.dict_column[col]["n_missing"] = missing_count
            self.dict_column[col]["p_missing"] = (missing_count /
                                                  self.n_samples) * 100
            self.dict_column[col]["n_unique"] = data[col].nunique()

        # Check for duplicate rows and update the dict for all columns
        is_duplicate = data.duplicated().any()
        for col in self.column_names:
            self.dict_column[col]["is_duplicate"] = is_duplicate

    def _calculate_correlations(self, data):

        # Numerical features
        corr = data[self.numerical_features + [self.target]].corr()
        self.df_corr = corr

    def distribution_variable(self, data, column, n_bins=None, bins=None):

        if column not in data.columns:
            raise ValueError("`column` must be a valid column name.")

        if n_bins is not None:
            min_value = data[column].min()
            max_value = data[column].max()
            bins = list(
                np.linspace(min_value, max_value + 1, n_bins + 1).astype(int))

        return pd.cut(data[column], bins=bins,
                      right=False).value_counts().sort_index()

    def plot_numerical_variables(self, data, num_vars=None):

        if num_vars is None:
            num_vars = self.numerical_features
        else:
            bad_input = []
            for x in num_vars:
                if x not in self.numerical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("The following columns are not numerical features: "
                                 f"{bad_input}")

        num_graphs = len(num_vars)

        # Configurar el estilo de Seaborn
        sns.set(style="whitegrid")

        # Caso de un solo gráfico
        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.histplot(data[num_vars[0]], bins=30, kde=True, color='skyblue')
            plt.title(f'{num_vars[0]} Distribution', fontsize=14, weight='bold')
            plt.xlabel(f'{num_vars[0]}')
            plt.ylabel('Frequency')

        else:
            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=(12, num_graphs * 2.5))

            for i, var in enumerate(num_vars):
                # Histograma
                sns.histplot(data[var],
                             bins=30,
                             kde=True,
                             ax=axes[i],
                             color='skyblue')
                # Añadir título
                axes[i].set_title(f'{var} Distribution',
                                  fontsize=14,
                                  weight='bold')
                # Añadir título al eje X
                axes[i].set_xlabel(f'{var}')
                # Añadir título al eje y
                axes[i].set_ylabel('Frequency')

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

        plt.tight_layout()
        plt.show()

    def plot_categorical_variables(self, data, cat_vars=None):

        if cat_vars is None:
            cat_vars = self.categorical_features
        else:
            bad_input = []
            for x in cat_vars:
                if x not in self.categorical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("The following columns are not categorical features: "
                                 f"{bad_input}")

        num_graphs = len(cat_vars)

        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.countplot(data, x=cat_vars[0], color='skyblue')
            plt.title(f'{cat_vars[0]} Distribution', fontsize=14, weight='bold')
            plt.xlabel(f'{cat_vars[0]}')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        else:
            n = 2
            vls = np.arange(0, num_graphs + num_graphs % 2, 1)
            layout = [vls[i: i + n] for i in range(0, num_graphs, n)]
            fig, axes = plt.subplot_mosaic(layout, figsize=(12, num_graphs * 2.5))

            for i, var in enumerate(cat_vars):
                sns.countplot(data, x=var, ax=axes[i], color='skyblue')
                axes[i].set_title(f'{var} Distribution',
                                  fontsize=14,
                                  weight='bold')
                axes[i].set_xlabel(f'{var}')
                axes[i].set_ylabel('Count')

                # Rotar las etiquetas del eje x
                axes[i].tick_params(axis='x', rotation=45)  # O prueba con rotation=90 si es necesario

            # Eliminar subplot en caso de que haya un número impar de subplots
            if num_graphs % 2 != 0:
                fig.delaxes(axes[num_graphs])

        plt.tight_layout()
        plt.show()

    def plot_target(self, data):

        plt.figure(figsize=(10, 6))
        sns.countplot(data=data,
                      x=self.target,
                      hue=self.target,
                      palette="pastel")
        plt.title('Distribution of Default')
        plt.xlabel('Default')
        plt.ylabel('Count')

    def plot_correlations(self):

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

        if num_vars is None:
            num_vars = self.numerical_features
        else:
            bad_input = []
            for x in num_vars:
                if x not in self.numerical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("The following columns are not numerical features: "
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

        if cat_vars is None:
            cat_vars = self.categorical_features
        else:
            bad_input = []
            for x in cat_vars:
                if x not in self.categorical_features:
                    bad_input.append(x)
            if len(bad_input) > 0:
                raise ValueError("The following columns are not categorical features: "
                                 f"{bad_input}")

        num_graphs = len(cat_vars)

        if num_graphs == 1:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=cat_vars[0], hue=self.target, data=data, palette='Blues_d',
                          order=data[cat_vars[0]].value_counts().index)
            plt.title(f'Distribución de {cat_vars[0]} por Default')
            plt.xlabel(cat_vars[0])
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
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
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].get_legend().remove(
                )  # Remover la leyenda de cada subgráfico
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

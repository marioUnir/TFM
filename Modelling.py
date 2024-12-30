import numpy as np
import pandas as pd
import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import joblib

class ModelSelectionPipeline:
    def __init__(
            self,
            test_size=0.2, 
            random_state=42,
            models=None,
            param_grids=None,
            save_path="outputs/"
        ):

        self.test_size = test_size
        self.random_state = random_state
        self.models = models
        self.param_grids = param_grids
        self.save_path = save_path

        # Seleccionamos los modelos de Machine Learning
        self._initialize_models()

        # Fijar hiperparámetros
        self._set_hyperparameters()

    def _split_data(self, X, y):

        print("División en train y test ...")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        return X_train, X_val, y_train, y_val
    
    def _apply_smote(self, X_train, y_train):
        
        print("Balanceando el dataset en train ...")
        
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        return X_train_smote, y_train_smote

    def _initialize_models(self):

        if self.models is None:
            self.models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
                "Naive Bayes": GaussianNB(),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                "Random Forest": RandomForestClassifier(random_state=self.random_state),
                "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state)
            }

        print(f"Estimadores seleccionados : {list(self.models.keys())}")

    def _set_hyperparameters(self):

        if self.param_grids is None:
            self.param_grids = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"],
                    "penalty": ["l2"]
                },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 4],
                    "min_samples_split": [50, 500]
                },
                "Naive Bayes": {
                    "var_smoothing": np.logspace(0, -9, num=5)
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5],
                    "min_samples_split": [50, 500]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }
            }
        else:
            if list(self.models.keys()) != list(self.param_grids.keys()):
                raise ValueError("`models` y `param_grids` no son coherentes. Revísalos.")

    def _fit_grid_search(self, X_train_smote, y_train_smote):

        print("Búsqueda de hiperparámetros ...")

        start = time.time()

        self._best_models = {}
        grid_results = []
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)

        for model_name, model in self.models.items():
            print(f"** {model_name} **")
            start_model = time.time()
            grid = GridSearchCV(
                estimator=model,
                param_grid=self.param_grids[model_name],
                cv=kfold,
                scoring="accuracy",
                n_jobs=-2
            )
            grid.fit(X_train_smote, y_train_smote)
            elapsed_time_model = time.time() - start_model
            self._best_models[model_name] = grid.best_estimator_
            grid_results.append({
                "Model": model_name,
                "Best Params": grid.best_params_,
                "Best Accuracy": grid.best_score_,
                "Execution Time (s)": elapsed_time_model
            })
        
        grid_results_df = pd.DataFrame(grid_results)
        grid_results_df.sort_values(by="Best Accuracy", ascending=False, inplace=True)
        self._grid_results_df = grid_results_df

        self.grid_search_time = time.time() - start
        if self.grid_search_time > 60:
            print(f"Tiempo total (min): {self.grid_search_time / 60:.2f}")
        else:
            print(f"Tiempo total (s): {self.grid_search_time:.2f}")

    def _evaluate_models(self, X_train_smote, y_train_smote):

        print("Evaluando mejor modelo ...")

        start = time.time()

        evaluation_summary = []
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)

        for model_name, best_model in self.best_models.items():
            print(f"** {model_name} **")
            start_model = time.time()
            accuracy = cross_val_score(best_model, X_train_smote, y_train_smote, cv=kfold, scoring="accuracy").mean()
            precision = cross_val_score(best_model, X_train_smote, y_train_smote, cv=kfold, scoring="precision_weighted").mean()
            recall = cross_val_score(best_model, X_train_smote, y_train_smote, cv=kfold, scoring="recall_weighted").mean()
            f1 = cross_val_score(best_model, X_train_smote, y_train_smote, cv=kfold, scoring="f1_weighted").mean()
            roc_auc = cross_val_score(best_model, X_train_smote, y_train_smote, cv=kfold, scoring="roc_auc").mean()
            elapsed_time_model = time.time() - start_model

            evaluation_summary.append({
                "Model": model_name,
                "Mean Accuracy": accuracy,
                "Mean Precision": precision,
                "Mean Recall": recall,
                "Mean F1-Score": f1,
                "Mean ROC-AUC": roc_auc,
                "Execution Time (s)": elapsed_time_model
            })

        self._evaluation_df = pd.DataFrame(evaluation_summary)
        self._evaluation_df.sort_values(by="Mean Accuracy", ascending=False, inplace=True)

        self.evaluation_time = time.time() - start
        if self.evaluation_time > 60:
            print(f"Tiempo total (min): {self.evaluation_time / 60:.2f}")
        else:
            print(f"Tiempo total (s): {self.evaluation_time:.2f}")

    def save_results(self, save_path=None):

        print("Guardando resultados ...")

        if save_path is None:
            save_path = "outputs/"
        else:
            if save_path[-1] != "/":
                save_path += "/"

        self._grid_results_df.to_excel(f"{save_path}grid_results_df.xlsx", index=False)
        self._evaluation_df.to_excel(f"{save_path}evaluation_df.xlsx", index=False)

        for model_name, best_model in self.best_models.items():
            filename = f"{save_path}{model_name.replace(' ', '_')}_best_model.joblib"
            joblib.dump(best_model, filename)

    def run(self, X, y):

        # Paso 1 : Dividir la muestra en train y test
        X_train, X_val, y_train, y_val = self._split_data(X, y)

        # Paso 2 : Balancear el dataset
        X_train_smote, y_train_smote = self._apply_smote(X_train=X_train, y_train=y_train)
              
        # Paso 3 : Búsqueda de hiperparámetros
        self._fit_grid_search(X_train_smote=X_train_smote, y_train_smote=y_train_smote)
        
        # Paso 4 : Evaluación de modelos
        self._evaluate_models(X_train_smote=X_train_smote, y_train_smote=y_train_smote)

        # Paso 5 : Guardamos los resultados
        self.save_results(save_path=self.save_path)

        return X_train, y_train, X_train_smote, y_train_smote, X_val, y_val
    
    @property
    def grid_results_df(self):

        if hasattr(self, "_grid_results_df") and isinstance(self._grid_results_df, pd.DataFrame):
            self._grid_results_df
        else:
            return self._load_grid_results_df_excel()
    
    @property
    def evaluation_df(self):

        if hasattr(self, "_evaluation_df") and isinstance(self._evaluation_df, pd.DataFrame):
            self._evaluation_df
        else:
            return self._load_evaluation_df_excel()
    
    @property
    def best_models(self):

        if hasattr(self, "_best_models") and isinstance(self._best_models, dict):
            return self._best_models
        else:
            return self._load_models(verbose=False)
    
    def _load_models(self, verbose=True):

        # Cargar los modelos desde los archivos y almacenarlos en best_models con los mismos nombres
        best_models = {}
        model_names = list(self.models.keys())

        # Iterar sobre los nombres de los modelos y cargar los archivos correspondientes
        for model_name in model_names:
            # Definir el nombre del archivo de cada modelo
            filename = f"{self.save_path}{model_name.replace(' ', '_')}_best_model.joblib"
            # Cargar el modelo y guardarlo en el diccionario best_models con el nombre original
            best_models[model_name] = joblib.load(filename)
            if verbose:
                print(f"Modelo {model_name} cargado desde {filename}")

        self._best_models = best_models

        return best_models
    
    def _load_evaluation_df_excel(self):

        self._evaluation_df = pd.read_excel(f"{self.save_path}evaluation_df.xlsx").set_index("Model")

        return self._evaluation_df
    
    def _load_grid_results_df_excel(self):

        self._grid_results_df = pd.read_excel(f"{self.save_path}grid_results_df.xlsx").set_index("Model")

        return self._grid_results_df
    
    def get_metrics_all_models(self, X, y):
        
        evaluation_summary = {}
        for model_name, best_model in self.best_models.items():
            
            # Realizamos predicciones
            y_pred = best_model.predict(X)
            y_pred_prob = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else None
            
            # Calculamos las métricas
            metrics = {
                'Accuracy': accuracy_score(y, y_pred),
                'Precision': precision_score(y, y_pred, zero_division=0),
                'Recall': recall_score(y, y_pred, zero_division=0),
                'F1-Score': f1_score(y, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y, y_pred_prob) if y_pred_prob is not None else None
            }
            evaluation_summary[model_name] = metrics

        # Convertir resultados de evaluación en un DataFrame
        metrics_df = pd.DataFrame(evaluation_summary).T
        metrics_df.sort_values(["Accuracy"], ascending=False, inplace=True)
        
        return metrics_df
    
    def get_metrics_by_class_all_models(self, X, y):

        # Función para calcular métricas usando scikit-learn
        def calculate_metrics_by_class(y_test, y_pred):
            # Calculamos la matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculamos métricas globales y por clase
            metrics = {
                'Confusion Matrix': cm,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision (0 - No Default)': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
                'Precision (1 - Default)': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                'Recall (0 - No Default)': recall_score(y_test, y_pred, pos_label=0, zero_division=0),
                'Recall (1 - Default)': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                'F1-Score (0 - No Default)': f1_score(y_test, y_pred, pos_label=0, zero_division=0),
                'F1-Score (1 - Default)': f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            }
            return metrics
        
        metrics_by_class = {}
        # Evaluamos cada modelo y almacenamos las métricas en el diccionario
        for model_name, best_model in self.best_models.items():
            
            y_pred = best_model.predict(X)
            
            # Calculamos las métricas y las añadimos al diccionario
            metrics_by_class[model_name] = calculate_metrics_by_class(y, y_pred)

        # Convertimos el diccionario de métricas a un DataFrame para facilitar la visualización
        metrics_df_by_class = pd.DataFrame(metrics_by_class).T

        # Mostramos las métricas
        metrics_df_by_class = metrics_df_by_class.drop(columns=['Confusion Matrix'])
        metrics_df_by_class.sort_values(["Accuracy"], ascending=False, inplace=True)
        metrics_df_by_class.drop(columns=["Accuracy"])

        self.metrics_df_by_class = metrics_df_by_class

        # Guardar matriz de confusión
        cm_model = {}
        for model_name, best_model in self.best_models.items():
            cm_model[model_name] = pd.DataFrame(metrics_by_class[model_name]["Confusion Matrix"])

        self.cm = cm_model

        return metrics_df_by_class

    @staticmethod
    def eval_model_test(model, X_test, y_test, classes=None):
        """
        Evalúa las métricas de un problema binario o multiclase dado un modelo y datos de prueba.

        Parámetros
        ----------
            model: El modelo que se está evaluando.
            X_test: Datos de características de prueba.
            y_test: Etiquetas verdaderas de prueba.
            classes (list, opcional): Lista de clases esperadas. Si es None, se asume [0, 1].
        
        Return
        ------
            dict: Diccionario con las métricas generales y por clase.
        """
        
        if classes is None:
            # Determinar clases únicas en y_test si no están proporcionadas
            classes = sorted(np.unique(y_test))

        # Predicción
        y_pred = model.predict(X_test)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        df_cm = pd.DataFrame(cm, columns=classes, index=classes)
        print("Matriz de confusión:")
        display(df_cm)
        
            # Total de instancias en el conjunto de prueba
        total_predictions = cm.sum()
        
        # Instancias clasificadas correctamente
        correct_predictions = sum(cm[i][i] for i in range(len(cm)))
        p_correct = round(correct_predictions / total_predictions * 100, 2)
        
        # Instancias clasificadas incorrectamente
        incorrect_predictions = total_predictions - correct_predictions
        p_incorrect = round(incorrect_predictions / total_predictions * 100, 2)

        print("\nInstancias clasificadas correctamente:", correct_predictions, "\n% Instancias clasificadas correctamente = Accuracy:", p_correct, "%")
    
        print("\nInstancias clasificadas incorrectamente:", incorrect_predictions, "\n% Instancias clasificadas incorrectamente = Error Rate:", p_incorrect, "%")
        
           
        # Métricas por clase
        metrics_by_class = {}
        for idx, category in enumerate(classes):
            total = cm.sum()
            positive = cm[idx, :].sum()
            negative = total - positive
            sum_col_i = cm[:, idx].sum()
        
            tp = cm[idx, idx]
            fn = positive - tp
            fp = sum_col_i - tp
            tn = total - positive - sum_col_i + tp

            precission = tp / (tp + fp)
            accuracy = (tp + tn) / (positive + negative)
            # recall
            tp_rate = tp / positive
            # specificity
            tn_rate = tn / negative
            fp_rate = fp / negative
            f1_score = (2 * precission * tp_rate) / (precission + tp_rate)
            support = sum(y_test == category)

            metrics_by_class[category] = {
                "Precision": precission,
                "Accuracy": accuracy,
                "Recall (Sensitivity)": tp_rate,
                "Specificity": tn_rate,
                "FP Rate": fp_rate,
                "F1-Score": f1_score,
                "Support": support
            }

        print(f"\nMétricas de cada clase:")
        metric_class = pd.DataFrame(metrics_by_class).transpose()
        display(metric_class)

        # Métricas generales
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = None
        if len(classes) == 2:
            # Solo calculamos ROC-AUC para problemas binarios
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Guardamos las métricas
        metrics = {
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc,
            "Metrics by Class": metric_class
        }

        # Métricas generales
        print("Métricas generales:\n")
        for key, value in metrics.items():
            if key != "Metrics by Class":
                print(f"{key}: {round(value, 4)}")

        return metrics
    
    def plot_evaluation_metrics(self):
        """
        Genera gráficos para las métricas de evaluación: Precision, Recall, F1-Score, y Accuracy.
        """

        if hasattr(self, "metrics_df_by_class"):
            metrics_df_by_class = self.metrics_df_by_class
        else:
            raise ValueError("Please execute first `get_metrics_by_class_all_models` method.")

        # Palette de tonos azules
        blue_palette = ["#1973B8", "#5BBEFF", "#004481"]

        # Función para añadir valores en las barras (evita anotar valores 0.0)
        def add_values_to_bars(ax):
            for bar in ax.patches:
                height = bar.get_height()
                if height > 0.01:  # Mostrar solo valores significativos
                    ax.annotate(f'{height:.2f}', 
                                (bar.get_x() + bar.get_width() / 2, height - 0.1),
                                ha='center', va='bottom', fontsize=12, color="white")

        # Métricas a graficar
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']

        # Crear subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Gráficos combinados para Precision, Recall y F1-Score
        for idx, metric in enumerate(metrics_to_plot):
            metric_0 = f'{metric} (0 - No Default)'
            metric_1 = f'{metric} (1 - Default)'
            
            data_to_plot = metrics_df_by_class[[metric_0, metric_1]].reset_index().melt(
                id_vars='index', var_name='Clase', value_name='Valor'
            )
            ax = sns.barplot(data=data_to_plot, x='index', y='Valor', hue='Clase', 
                            palette=blue_palette[:2], ax=axes[idx+1])
            add_values_to_bars(ax)
            ax.set_title(f'{metric} por Clase', fontsize=14)
            ax.set_xlabel('Modelo', fontsize=12)
            ax.set_ylabel(f'{metric}', fontsize=12)
            ax.set_xticks(range(len(ax.get_xticklabels())))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.get_legend().remove()  # Eliminamos la leyenda individual

        # Gráfico de Accuracy
        ax = sns.barplot(data=metrics_df_by_class.reset_index(), x='index', y='Accuracy', color=blue_palette[2], ax=axes[0])
        add_values_to_bars(ax)
        ax.set_title('Accuracy de Modelos', fontsize=14)
        ax.set_xlabel('Modelo', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xticks(range(len(ax.get_xticklabels())))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Crear una leyenda común fuera del gráfico
        handles = [
            Patch(color=blue_palette[0], label='No Default'),
            Patch(color=blue_palette[1], label='Default')
        ]
        fig.legend(handles=handles, title='Clase', loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)

        # Ajustar diseño
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Dejar espacio para la leyenda
        plt.show()

    def plot_roc_curve(self, X, y):

        # Crear una paleta de colores azules
        # colors = sns.color_palette("Blues", n_colors=len(self.best_models))

        # Crear un gráfico de Curvas ROC
        plt.figure(figsize=(10, 8))

        # Evaluar cada modelo en best_models
        for idx, (model_name, model) in enumerate(self.best_models.items()): 
            # Predecir probabilidades (para calcular la curva ROC)
            if hasattr(model, "predict_proba"):
                y_pred_prob = model.predict_proba(X)[:, 1]  # Probabilidades de la clase 1
            else:
                # Si no hay `predict_proba`, usar `decision_function` (ej. SVM)
                y_pred_prob = model.decision_function(X)
            
            # Calcular la curva ROC
            fpr, tpr, _ = roc_curve(y, y_pred_prob)
            auc = roc_auc_score(y, y_pred_prob)
            
            # Graficar con colores azules
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')#, color=colors[idx])

        # Graficar línea de referencia (clasificador aleatorio)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

        # Configuración del gráfico
        plt.title('Curvas ROC de los Modelos', fontsize=16)
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid()
        plt.tight_layout()
        plt.show()


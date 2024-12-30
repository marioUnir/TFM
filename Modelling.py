import numpy as np
import pandas as pd
import time

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

        self.best_models = {}
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
            self.best_models[model_name] = grid.best_estimator_
            grid_results.append({
                "Model": model_name,
                "Best Params": grid.best_params_,
                "Best Accuracy": grid.best_score_,
                "Execution Time (s)": elapsed_time_model
            })
        
        grid_results_df = pd.DataFrame(grid_results)
        grid_results_df.sort_values(by="Best Accuracy", ascending=False, inplace=True)
        self.grid_results_df = grid_results_df

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

        self.evaluation_df = pd.DataFrame(evaluation_summary)
        self.evaluation_df.sort_values(by="Mean Accuracy", ascending=False, inplace=True)

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

        self.grid_results_df.to_excel(f"{save_path}grid_results_df.xlsx", index=False)
        self.evaluation_df.to_excel(f"{save_path}evaluation_df.xlsx", index=False)

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

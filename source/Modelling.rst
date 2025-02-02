Modelling
=========

.. automodule:: Modelling
   :members:
   :undoc-members:
   :show-inheritance:

Example
~~~~~~~

>>> from Modelling import ModelSelectionPipeline
>>> pipeline = ModelSelectionPipeline(
>>>     test_size=0.2, 
>>>     random_state=42,
>>>     models=None,
>>>     param_grids=None
>>> )
Estimadores seleccionados : ['Logistic Regression', 'Decision Tree', 'Naive Bayes', 'K-Nearest Neighbors', 'Random Forest', 'Gradient Boosting']

>>> pipeline.models
{'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
 'Decision Tree': DecisionTreeClassifier(random_state=42),
 'Naive Bayes': GaussianNB(),
 'K-Nearest Neighbors': KNeighborsClassifier(),
 'Random Forest': RandomForestClassifier(random_state=42),
 'Gradient Boosting': GradientBoostingClassifier(random_state=42)}

>>> model_names = list(pipeline.models.keys())

>>> pipeline.param_grids
{'Logistic Regression': {'C': [0.1, 1, 10],
  'solver': ['lbfgs', 'liblinear'],
  'penalty': ['l2']},
 'Decision Tree': {'criterion': ['gini', 'entropy'],
  'max_depth': [3, 4],
  'min_samples_split': [50, 500]},
 'Naive Bayes': {'var_smoothing': array([1.00000000e+00, 5.62341325e-03, 3.16227766e-05, 1.77827941e-07,
         1.00000000e-09])},
 'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7],
  'weights': ['uniform', 'distance'],
  'metric': ['euclidean', 'manhattan']},
 'Random Forest': {'n_estimators': [50, 100],
  'max_depth': [3, 5],
  'min_samples_split': [50, 500]},
 'Gradient Boosting': {'n_estimators': [50, 100],
  'learning_rate': [0.01, 0.1],
  'max_depth': [3, 5]}}

>>> X_train, y_train, X_train_smote, y_train_smote, X_val, y_val = pipeline.run(X_scaled, y)
División en train y test ...
Balanceando el dataset en train ...
Búsqueda de hiperparámetros ...
** Logistic Regression **
** Decision Tree **
** Naive Bayes **
** K-Nearest Neighbors **
** Random Forest **
** Gradient Boosting **
Tiempo total (min): 8.54
Evaluando mejor modelo ...
** Logistic Regression **
** Decision Tree **
** Naive Bayes **
** K-Nearest Neighbors **
** Random Forest **
** Gradient Boosting **
Tiempo total (min): 28.95
Guardando resultados ...

Si ya hemos ejecutado y solo queremos obtener los resultados no es necesario volver a ejecutar

>>> # Paso 1 : Dividir la muestra en train y test
>>> X_train, X_val, y_train, y_val = pipeline._split_data(X_scaled, y)

>>> # Paso 2 : Balancear el dataset
>>> X_train_smote, y_train_smote = pipeline._apply_smote(X_train=X_train, y_train=y_train)

>>> pipeline.best_models
{'Logistic Regression': LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='liblinear'),
 'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_split=50, random_state=42),
 'Naive Bayes': GaussianNB(var_smoothing=3.1622776601683795e-05),
 'K-Nearest Neighbors': KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance'),
 'Random Forest': RandomForestClassifier(max_depth=5, min_samples_split=50, random_state=42),
 'Gradient Boosting': GradientBoostingClassifier(max_depth=5, random_state=42)}

 >>> pipeline.grid_results_df

===================  ================================================================  ===============  ====================
Model                Best Params                                                         Best Accuracy    Execution Time (s)
===================  ================================================================  ===============  ====================
Gradient Boosting    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}              0.837489             336.355
K-Nearest Neighbors  {'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}         0.830908              87.763
Random Forest        {'max_depth': 5, 'min_samples_split': 50, 'n_estimators': 100}           0.726124              46.4241
Logistic Regression  {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear'}                       0.708603              26.8873
Naive Bayes          {'var_smoothing': 3.1622776601683795e-05}                                0.694264               5.49103
Decision Tree        {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 50}           0.691616               9.5826
===================  ================================================================  ===============  ====================

>>> pipeline.evaluation_df

===================  ===============  ================  =============  ===============  ==============  ====================
Model                  Mean Accuracy    Mean Precision    Mean Recall    Mean F1-Score    Mean ROC-AUC    Execution Time (s)
===================  ===============  ================  =============  ===============  ==============  ====================
Gradient Boosting           0.837489          0.840906       0.837489         0.83708         0.907686            1477.79
K-Nearest Neighbors         0.830988          0.849635       0.830988         0.828698        0.902047              75.2865
Random Forest               0.726124          0.737461       0.726124         0.722819        0.798277             152.565
Logistic Regression         0.708603          0.720019       0.708603         0.704775        0.777219              14.0813
Naive Bayes                 0.694264          0.719754       0.694264         0.685134        0.763673               3.72403
Decision Tree               0.691616          0.729007       0.691616         0.678536        0.755418              13.4123
===================  ===============  ================  =============  ===============  ==============  ====================

Notebook
~~~~~~~~

.. nbgallery::

   notebooks/4-Modelling.ipynb
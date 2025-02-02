Preprocessing
=============

.. automodule:: Preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Example
~~~~~~~

>>> from Preprocessing import Preprocessing
>>> preprocessing = Preprocessing(
>>>    target=target,
>>>    missing_strategy_cat="most_frequent",
>>>    categorical_encoding="onehot",
>>>    scaler_method="minmax"
>>> )

>>> data = preprocessing.fit_transform(df)
Imputando valores ...
Codificando variables categóricas ...
Escalando variables numéricas ...

Notebook
~~~~~~~~

.. nbgallery::

   notebooks/3-Preprocessing.ipynb
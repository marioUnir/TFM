Outliers
========

.. automodule:: Outliers
   :members:
   :undoc-members:
   :show-inheritance:

Example
~~~~~~~

>>> from Outliers import OutliersDetection
>>> outliers = OutliersDetection(
>>>     method="all",
>>>     k=1.5
>>> )

>>> outliers.run(df)
** Variables Numéricas **
Ejecutando método IQR...
Ejecutando método std...
Ejecutando método IForest...
Ejecutando método LOF...
Ejecutando método GMM...
** Variables Categóricas **

>>> resultsIQR = outliers.get_results(method="IQR")
>>> resultsIQR

==================  ========  =========  =========  ==========  =========  ============  ============
..                        Q1         Q3        IQR          lb         ub    n_outliers    p_outliers
==================  ========  =========  =========  ==========  =========  ============  ============
CreditLimit         50000     240000     190000     -235000     525000              167      0.556667
Age                    28         41         13           8.5       60.5            272      0.906667
BillAmountSep        3558.75   67091      63532.2    -91739.6   162389             2400      8
BillAmountAug        2984.75   64006.2    61021.5    -88547.5   155538             2395      7.98333
BillAmountJul        2666.25   60164.8    57498.5    -83581.5   146412             2469      8.23
BillAmountJun        2326.75   54506      52179.2    -75942.1   132775             2622      8.74
BillAmountMay        1763      50190.5    48427.5    -70878.2   122832             2725      9.08333
BillAmountApr        1256      49198.2    47942.2    -70657.4   121112             2693      8.97667
PreviousPaymentSep   1000       5006       4006       -5009      11015             2745      9.15
PreviousPaymentAug    833       5000       4167       -5417.5    11250.5           2714      9.04667
PreviousPaymentJul    390       4505       4115       -5782.5    10677.5           2598      8.66
PreviousPaymentJun    296       4013.25    3717.25    -5279.88    9589.12          2994      9.98
PreviousPaymentMay    252.5     4031.5     3779       -5416       9700             2945      9.81667
PreviousPaymentApr    117.75    4000       3882.25    -5705.62    9823.38          2958      9.86
==================  ========  =========  =========  ==========  =========  ============  ============

>>> results = outliers.get_results()
>>> results

==================  ================  ================  ================  ================  ====================  ====================  ================  ================  ================  ================
..                    IQR n_outliers    IQR p_outliers    std n_outliers    std p_outliers    IForest n_outliers    IForest p_outliers    LOF n_outliers    LOF p_outliers    GMM n_outliers    GMM p_outliers
==================  ================  ================  ================  ================  ====================  ====================  ================  ================  ================  ================
CreditLimit                      167          0.556667              2476           8.25333                  1497               4.99                 1497           4.99                 1497           4.99
Age                              272          0.906667              2747           9.15667                  1120               3.73333              1120           3.73333              1120           3.73333
BillAmountSep                   2400          8                     2415           8.05                     1494               4.98                 1494           4.98                 1494           4.98
BillAmountAug                   2395          7.98333               2386           7.95333                  1495               4.98333              1495           4.98333              1495           4.98333
BillAmountJul                   2469          8.23                  2337           7.79                     1496               4.98667              1496           4.98667              1496           4.98667
BillAmountJun                   2622          8.74                  2365           7.88333                  1490               4.96667              1490           4.96667              1490           4.96667
BillAmountMay                   2725          9.08333               2436           8.12                     1499               4.99667              1499           4.99667              1499           4.99667
BillAmountApr                   2693          8.97667               2446           8.15333                  1495               4.98333              1495           4.98333              1495           4.98333
PreviousPaymentSep              2745          9.15                   786           2.62                     1500               5                    1500           5                    1500           5
PreviousPaymentAug              2714          9.04667                606           2.02                     1500               5                    1500           5                    1500           5
PreviousPaymentJul              2598          8.66                   728           2.42667                  1413               4.71                 1413           4.71                 1413           4.71
PreviousPaymentJun              2994          9.98                   858           2.86                     1493               4.97667              1493           4.97667              1493           4.97667
PreviousPaymentMay              2945          9.81667                840           2.8                      1491               4.97                 1491           4.97                 1491           4.97
PreviousPaymentApr              2958          9.86                   803           2.67667                  1498               4.99333              1498           4.99333              1498           4.99333
==================  ================  ================  ================  ================  ====================  ====================  ================  ================  ================  ================

>>> data_transformed = outliers.transform(df, method="IQR", metric="median")

>>> results_cat = outliers.get_results_cat_vars()
>>> results_cat

==================  ============  ============  =============================================================================================================================
..                    n_outliers    p_outliers  categories
==================  ============  ============  =============================================================================================================================
Gender                         0       0        nan
EducationLevel               454       1.51333  ['Others', 'Unknown']
Marriage                     377       1.25667  ['Others', 'Unknown']
RepaymentStatusSep           463       1.54333  ['Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
RepaymentStatusAug           511       1.70333  ['Delay 1 Month', 'Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
RepaymentStatusJul           394       1.31333  ['Delay 1 Month', 'Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
RepaymentStatusJun           351       1.17     ['Delay 1 Month', 'Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
RepaymentStatusMay           342       1.14     ['Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
RepaymentStatusApr           313       1.04333  ['Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
Default                        0       0        nan
==================  ============  ============  =============================================================================================================================

 Combinamos algunas categorías en una sola puesto que tienen poca representatividad

>>> lst_vars_combine = ['RepaymentStatusSep', 'RepaymentStatusAug', 'RepaymentStatusJul', 'RepaymentStatusJun', 'RepaymentStatusMay', 'RepaymentStatusApr']
>>> categories_combine = ['Delay 3 Months', 'Delay 4 Months', 'Delay 5 Months', 'Delay 6 Months', 'Delay 7 Months', 'Delay 8 Months']
>>> for var in lst_vars_combine:
>>>     data_transformed = outliers.combine_categories(data_transformed, var, categories_combine, "Delay 2+ Months")

Notebook
~~~~~~~~

.. nbgallery::

   notebooks/2-Outliers.ipynb
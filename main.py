print("hello World")

print("1")

import pandas as pd
import numpy as np

from codigos.EDA import EDAProcessor

# Load the dataset
data = pd.read_csv('data/UCI_Credit_Card.csv')
print(data.head())
print(type(data))

# Preprocessing columns
# Cambiamos el nombre de las columnas por uno más descriptivo

descriptive_columns = {
    'ID': 'Id',
    'LIMIT_BAL': 'CreditLimit',
    'SEX': 'Gender',
    'EDUCATION': 'EducationLevel',
    'MARRIAGE': 'Marriage',
    'AGE': 'Age',
    'PAY_0': 'RepaymentStatusSep',
    'PAY_2': 'RepaymentStatusAug',
    'PAY_3': 'RepaymentStatusJul',
    'PAY_4': 'RepaymentStatusJun',
    'PAY_5': 'RepaymentStatusMay',
    'PAY_6': 'RepaymentStatusApr',
    'BILL_AMT1': 'BillAmountSep',
    'BILL_AMT2': 'BillAmountAug',
    'BILL_AMT3': 'BillAmountJul',
    'BILL_AMT4': 'BillAmountJun',
    'BILL_AMT5': 'BillAmountMay',
    'BILL_AMT6': 'BillAmountApr',
    'PAY_AMT1': 'PreviousPaymentSep',
    'PAY_AMT2': 'PreviousPaymentAug',
    'PAY_AMT3': 'PreviousPaymentJul',
    'PAY_AMT4': 'PreviousPaymentJun',
    'PAY_AMT5': 'PreviousPaymentMay',
    'PAY_AMT6': 'PreviousPaymentApr',
    'default.payment.next.month': 'Default'
}

data.rename(columns=descriptive_columns, inplace=True)

# Convert categorical variables with their appropriate labels based on the provided descriptions
data['Gender'] = data['Gender'].map({1: 'Male', 2: 'Female'})
data['EducationLevel'] = data['EducationLevel'].replace({
    0: np.nan,
    1: 'Graduate School',
    2: 'University',
    3: 'High School',
    4: 'Others',
    5: 'Unknown',
    6: 'Unknown'
})
data['Marriage'] = data['Marriage'].replace({
    1: 'Married',
    2: 'Single',
    3: 'Others',
    0: 'Unknown'
})

# For repayment status, values range from -1 (paid duly) to 9 (severe delay)
repayment_status_mapping = {
    -2: np.nan,
    -1: 'Paid Duly',
    0: 'No Consumption',
    1: 'Delay 1 Month',
    2: 'Delay 2 Months',
    3: 'Delay 3 Months',
    4: 'Delay 4 Months',
    5: 'Delay 5 Months',
    6: 'Delay 6 Months',
    7: 'Delay 7 Months',
    8: 'Delay 8 Months',
    9: 'Delay 9+ Months'
}

# Applying this mapping to all repayment status columns
repayment_columns = [x for x in data.columns if 'RepaymentStatus' in x]
for col in repayment_columns:
  data[col] = data[col].map(repayment_status_mapping)

# Parameters
target = "Default"

eda = EDAProcessor(target=target,
                   ordinal_columns=["EducationLevel"],
                   exclude_columns=["Id"])
eda.run(data)

print(data.dtypes)

print(eda.distribution_variable(data, "Age", n_bins=10))
print(eda.distribution_variable(data, "CreditLimit", n_bins=10))

# eda.plot_numerical_variables(data, n_rows=2, n_cols=2)
# eda.plot_target(data)

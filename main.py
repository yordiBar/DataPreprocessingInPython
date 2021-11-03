import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')

# matrix of features -> x
# iloc function = locate indexes, : = range (all rows), :-1 all columns except last one
# features in first column and dependant variable vector in last column
x = dataset.iloc[:, :-1].values

# dependant variable vector -> y
y = dataset.iloc[:, -1].values

# replace missing values with an average of column values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# print(x)

# import data transformation package
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# Feature scaling using standardisation technique
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print(x_train)
print(x_test)
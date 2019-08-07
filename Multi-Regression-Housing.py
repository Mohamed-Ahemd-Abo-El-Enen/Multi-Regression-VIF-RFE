import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def normalize(x):
    return ((x - np.min(x)) / (max(x) - min(x)))


housing = pd.read_csv("data/Housing.csv")

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})
housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})
housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})
housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})
housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})
housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})
#print(housing.head())
status = pd.get_dummies(housing["furnishingstatus"], drop_first=True)
#print(status)

housing = pd.concat([housing, status], axis=1)

#print(housing.head())


housing["areaperbedroom"] = housing["area"] / housing["bedrooms"]
housing["bathroomratio"] = housing["bathrooms"] / housing["bedrooms"]

housing.drop(['furnishingstatus'], axis=1, inplace=True)

x = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'semi-furnished', 'unfurnished',
       'areaperbedroom', 'bathroomratio']]

x = housing.apply(normalize)

y = housing['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)
x_train = sm.add_constant(x_train)
lm_1 = sm.OLS(y_train, x_train).fit()
print(lm_1.summary())


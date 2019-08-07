import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def vif_cal(input_data, dependent_col):
    vif_df = pd.DataFrame(columns=['Var', 'VIF'])
    x_vars = input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0, xvar_names.shape[0]):
        y = x_vars[xvar_names[i]]
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.OLS(y, x).fit().rsquared
        vif = round(1 / (1-rsq), 2)
        vif_df.loc[i] = [xvar_names[i], vif]
    return vif_df.sort_values(by='VIF', axis=0, ascending=False, inplace=False)


def normalize(x):
    return ((x - np.min(x)) / (max(x) - min(x)))


housing = pd.read_csv("data/Housing.csv")

housing['mainroad'] = housing['mainroad'].map({'yes': 1, 'no': 0})
housing['guestroom'] = housing['guestroom'].map({'yes': 1, 'no': 0})
housing['basement'] = housing['basement'].map({'yes': 1, 'no': 0})
housing['hotwaterheating'] = housing['hotwaterheating'].map({'yes': 1, 'no': 0})
housing['airconditioning'] = housing['airconditioning'].map({'yes': 1, 'no': 0})
housing['prefarea'] = housing['prefarea'].map({'yes': 1, 'no': 0})

status = pd.get_dummies(housing["furnishingstatus"], drop_first=True)

housing = pd.concat([housing, status], axis=1)

housing["areaperbedroom"] = housing["area"] / housing["bedrooms"]
housing["bathroomratio"] = housing["bathrooms"] / housing["bedrooms"]

housing.drop(['furnishingstatus'], axis=1, inplace=True)

housing = housing.apply(normalize)

x = housing[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'semi-furnished', 'unfurnished',
       'areaperbedroom', 'bathroomratio']]

print(vif_cal(input_data=housing, dependent_col='price'))
plt.figure(figsize=(16, 10))
sns.heatmap(housing.corr(), annot=True)
plt.show()


x = x.drop('bathroomratio', 1)
y = housing['price']
lm_1 = sm.OLS(y, x).fit()
print(lm_1.summary())

x = x.drop('bedrooms', 1)
lm_2 = sm.OLS(y, x).fit()
print(lm_2.summary())

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)
lm_3 = sm.OLS(y_train, x_train).fit()
y_predict = lm_3.predict(x_test)
print(y_predict)

c = [i for i in range(1, 165, 1)]
fig = plt.figure()
plt.plot(c, y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c, y_predict, color="red", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("sales", fontsize=16)
plt.show()


c = [i for i in range(1, 165, 1)]
fig = plt.figure()
plt.plot(c, y_test-y_predict, color="blue", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("y - y hate", fontsize=16)
plt.show()

mse = mean_squared_error(y_test, y_predict)
r_squared = r2_score(y_test, y_predict)
print("MSE : ", mse)
print("R square value : ", r_squared)

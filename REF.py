import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def normalize(x):
    return ((x - np.min(x)) / (max(x) - min(x)))

def VIF_Cal(input_data, dependent_col):
    vif_df = pd.DataFrame(columns=["VAR", "VIF"])
    x_vars = input_data.drop([dependent_col], axis=1)
    xvar_names = x_vars.columns
    for i in range(0, xvar_names.shape[0]):
        y = x_vars[xvar_names[i]]
        x = x_vars[xvar_names.drop(xvar_names[i])]
        rsq = sm.OLS(y, x).fit().rsquared
        vif = round(1/(1-rsq), 2)
        vif_df.loc[i] = [xvar_names[i], vif]

    return vif_df.sort_values(by="VIF", axis=0, ascending=False, inplace=False)


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

y = housing['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=100)

lrm = LinearRegression()
rfe = RFE(lrm, 9)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

col = x_train.columns[rfe.support_]

x_train_ref = x_train[col]

x_train_ref = sm.add_constant(x_train_ref)

lrm_rfe = sm.OLS(y_train, x_train_ref).fit()

print(lrm_rfe.summary())



#-------------------VIF

vif = VIF_Cal(input_data=housing.drop(["area", "bedrooms", "stories", "basement", "semi-furnished", "areaperbedroom"],
                                      axis=1),
              dependent_col="price")
print(vif)
x_test_vif = x_test[col]

x_train_vif = sm.add_constant(x_train[col])
x_test_vif = sm.add_constant(x_test_vif)

lrm_vif = sm.OLS(y_train, x_train_vif).fit()

y_predict_vif =lrm_vif.predict(x_test_vif)

c = [i for i in range(1, 165, 1)]
fig = plt.figure()
plt.plot(c, y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c, y_predict_vif, color="red", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("sales", fontsize=16)
plt.show()


c = [i for i in range(1, 165, 1)]
fig = plt.figure()
plt.plot(c, y_test-y_predict_vif, color="blue", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("y - y hate", fontsize=16)
plt.show()

mse = mean_squared_error(y_test, y_predict_vif)
r_squared = r2_score(y_test, y_predict_vif)
print("MSE : ", mse)
print("R square value : ", r_squared)


plt.scatter(y_test, y_predict_vif)
plt.suptitle("y-test vs y-pred", fontsize=10)
plt.xlabel("y-test", fontsize=18)
plt.ylabel("y-pred", fontsize=16)
plt.show()


sns.distplot((y_test-y_predict_vif), bins=50)
plt.suptitle("ERROR TERM", fontsize=20)
plt.xlabel("y-test - y-pred", fontsize=10)
plt.ylabel("Index", fontsize=16)
plt.show()

print("RMS : ", np.sqrt(mean_squared_error(y_test, y_predict_vif)))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

advertising = pd.read_csv("data/advertising.csv")
#print(advertising.head())
#print(advertising.tail())
#print(advertising.info())
#print(advertising.describe())
#sns.pairplot(advertising, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales", size=7)
#plt.show()

X = advertising[["TV", "Radio", "Newspaper"]]
Y = advertising["Sales"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=100)

mlr = LinearRegression()
mlr.fit(X_train, Y_train)
print(mlr.intercept_)
coeff_df = pd.DataFrame(mlr.coef_, index=X_test.columns, columns=["Coefficient"])
print(coeff_df)

y_predict = mlr.predict(X_test)
mse = mean_squared_error(Y_test, y_predict)
r_squared = r2_score(Y_test, y_predict)
print("MSE : ", mse)
print("R square value : ", r_squared)

X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)
mlr2 = sm.OLS(Y_train, X_train_sm).fit()
print(mlr2.params)
print(mlr2.summary())

plt.figure(figsize=(5, 5))
sns.heatmap(advertising.corr(), annot=True)
plt.show()

x_train_new = X_train[["TV", "Radio"]]
x_test_new = X_test[["TV", "Radio"]]

mlr.fit(x_train_new, Y_train)
y_predict_new = mlr.predict(x_test_new)

c = [i for i in range(1, 61, 1)]
fig = plt.figure()
plt.plot(c, Y_test, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c, y_predict_new, color="red", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("sales", fontsize=16)
plt.show()


c = [i for i in range(1, 61, 1)]
fig = plt.figure()
plt.plot(c, Y_test-y_predict_new, color="blue", linewidth=2.5, linestyle="-")
plt.xlabel("index", fontsize=18)
plt.ylabel("y - y hate", fontsize=16)
plt.show()

mse = mean_squared_error(Y_test, y_predict_new)
r_squared = r2_score(Y_test, y_predict_new)
print("MSE : ", mse)
print("R square value : ", r_squared)

x_train_final = x_train_new
x_train_final = sm.add_constant(x_train_final)
mlr2_final = sm.OLS(Y_train, x_train_final).fit()
print(mlr2_final.summary())


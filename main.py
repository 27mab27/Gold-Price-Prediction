import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("Gold Price (2013-2023).csv")

#
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df['Date_year']=df['Date'].dt.year
df['Date_month']=df['Date'].dt.month
df['Date_day']=df['Date'].dt.day
print(df.info())


#
columns=["Price","Open","High","Low"]
for col in columns:
    df[col] = df[col].replace({",":""}, regex=True)
    df[col] = df[col].astype("float64")


# Replace 'K' with '000' (assuming 'K' stands for thousand)
df['Vol.'] = df['Vol.'].str.replace("K", "000", regex=True)

# Remove the '%' sign from 'Change %'
df['Change %'] = df['Change %'].str.replace("%", "", regex=True)

# Convert these columns to float64
df['Vol.'] = df['Vol.'].astype("float64")
df['Change %'] = df['Change %'].astype("float64")

#
print("number of null is: ",df.isna().sum())
df = df.dropna(subset=['Vol.'])
#

"""""
sns.boxplot(x=df["Open"])
plt.show()
sns.boxplot(x=df["High"])
plt.show()
sns.boxplot(x=df["Low"])
plt.show()
sns.boxplot(x=df["Change %"])
plt.show()
sns.boxplot(x=df['Vol.'])
plt.show()
"""


columns=["Open","High","Low",'Vol.',"Change %"]
plt.show()
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(len(df[(df[col] > upper_bound) | (df[col] < lower_bound)]))
    df = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]


""""
fig = plt.figure(figsize=(10,6))
plt.scatter(y=df["Price"],x=df["Date"],color='red', marker='+')
plt.show()
"""

col = df.columns.drop(['Price','Date'])
x=df[col]
y= df["Price"]
print(x.head())
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.30)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Setting up the range of alpha values to test
alpha_values = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Initialize Ridge Regression
ridge = Ridge()

# Setup GridSearchCV
grid = GridSearchCV(estimator=ridge, param_grid=alpha_values, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid.fit(X_train, y_train)

print()
# Finding the best parameters
print("Best parameters found: ", grid.best_params_)
print("Best cross-validation score (negative MSE): ", grid.best_score_)

# Using the best parameters found by GridSearchCV
ridge_best = grid.best_estimator_

# Predicting with the best estimator
y_pred = ridge_best.predict(X_test)

fig = plt.figure(figsize=(10,6))

sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test,y= y_test)

plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Prediced  count', fontsize=14)
plt.title('Actual vs Predicted  count', fontsize=17)
plt.show()


print("Linear Regression: ")
print("------------------------------------------------ ")

# calculate R2 using scikit-learn
print("R2: ",r2_score(y_test,y_pred))

# calculate RMSE using scikit-learn
print("Mean Squared Error: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# calculate MSE using scikit-learn
print("Root Mean Squared Error: ",mean_squared_error(y_test,y_pred))

rando = RandomForestRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Setup the GridSearchCV
grid_search = GridSearchCV(estimator=rando, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
print()
print("Best parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)



print("Random Forest Regressor: ")
print("------------------------------------------------ ")
# calculate R2 using scikit-learn
print("R2: ",r2_score(y_test,y_pred))

# calculate RMSE using scikit-learn
print("Mean Squared Error: ",np.sqrt(mean_squared_error(y_test,y_pred)))

# calculate MSE using scikit-learn
print("Root Mean Squared Error: ",mean_squared_error(y_test,y_pred))
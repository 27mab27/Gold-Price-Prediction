import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Gold Price (2013-2023).csv")
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)

#
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(by='Date', ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

#
columns=["Price","Open","High","Low"]
for col in columns:
    df[col] = df[col].replace({",":""}, regex=True)
    df[col] = df[col].astype("float64")

#
print(df.isna().sum())

#
sns.boxplot(x=df["Open"])
plt.show()
sns.boxplot(x=df["High"])
plt.show()
sns.boxplot(x=df["Low"])
plt.show()

columns=["Open","High","Low"]
plt.show()
for col in columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(len(df[(df[col] > upper_bound) | (df[col] < lower_bound)]))
    df = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]


fig = plt.figure(figsize=(10,6))
plt.scatter(y=df["Price"],x=df["Date"],color='red', marker='+')
plt.show()

col = df.columns.drop(['Price','Date'])
x=df[col]
y= df["Price"]
print(x.head())
print(y.head())

# Create linear regression object
reg = LinearRegression()
reg.fit(x,y)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.30)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# instantiate
linreg = LinearRegression()
# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)
fig = plt.figure(figsize=(10,6))

sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
sns.lineplot(x=y_test,y= y_test)

plt.xlabel('Actual count', fontsize=14)
plt.ylabel('Prediced  count', fontsize=14)
plt.title('Actual vs Predicted  count', fontsize=17)
plt.show()

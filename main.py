import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


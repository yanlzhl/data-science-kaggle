import pandas as pd
import numpy as np


data=pd.read_csv('../../data/breast-cancer-wisconsin.csv')
df=pd.DataFrame(data)

# print(type(data))

# print(df.head())
print(df.columns)
# print(df.columns[-2])
# print(df.shape)
# print(df.describe())
# print(df['diagnosis'])
print(data.shape)

column_names = []

# print(data.describe())
# replace with standard empty value
df2=df.drop('Unnamed: 32',axis=1)
data =  df2.replace(to_replace='?',value=np.nan)
data = df2.dropna(how='any')
print(data.shape)

# using train_test_split to split the date to train and test
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data['data'], data[data[]], test_size=0, random_state=33)

from sklearn.preprocessing import StandardScaler

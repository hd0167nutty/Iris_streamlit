# Library
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Read Datasets
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Target
df['target'] = iris.target

# 目標値を数字から花の名前に変更
# Change the target value from number to a flower name
df.loc[df['target'] == 0, 'target'] = 'setosa'
df.loc[df['target'] == 1, 'target'] = 'versicolor'
df.loc[df['target'] == 2, 'target'] = 'virginica'

# 予測モデル構築
# Predicting model
x = iris.data[:, [0, 2]]
y = iris.target

# ロジスティック回帰
# Logistic regression
clf = LogisticRegression()
clf.fit(x, y)

# Sidebar(Input)
st.sidebar.header('Input Features')

sepalValue = st.sidebar.slider('sepal length(cm)', min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider('petal length(cm)', min_value=0.0, max_value=10.0, step=0.1)

# Main Panel
st.title('Iris Classifier')
st.write('## Imput Value')

# Input Data(1 row DataFrame)
value_df = pd.DataFrame([], columns=['data', 'sepal length(cm)', 'petal length(cm)'])
record = pd.Series(['data', sepalValue, petalValue], index=value_df.columns)
value_df = pd.concat([value_df, pd.DataFrame([record])], ignore_index=True, axis=0)
value_df.set_index('data', inplace=True)

# Input value
st.write(value_df)

# DataFrame of predicted value
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# Output predicted result
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です!')



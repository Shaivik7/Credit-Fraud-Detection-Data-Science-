#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:31:39 2023

@author: shaivik7
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style='darkgrid', palette='crest')


file = "creditcard.csv"
load_fraud_data = open(file,'rt')

#Loading the data onto a dataframe
fraud_data = pd.read_csv(load_fraud_data)

#Peeking at the data
print(fraud_data.head())
print(fraud_data.info())
print(fraud_data.shape)
fraud_data['Class'] = fraud_data['Class'].astype('category')
fraud_data['Hour'] = fraud_data['Time'].apply(lambda x: int(np.ceil(float(x)/3600) % 24))

#Data visualization
fig, axs = plt.subplots(nrows=2, figsize=(20,8))
axs = axs.ravel()

axs[0].set_title("Non-Fraudulent Transaction")
sns.countplot(data=fraud_data[ fraud_data.Class == 0 ], x= 'Hour', ax=axs[0])

axs[1].set_title("Fraudulent Transaction")
sns.countplot(data=fraud_data[ fraud_data.Class == 1 ], x= 'Hour', ax=axs[1])

plt.tight_layout()

#Correlation
fraud_data = fraud_data.drop('Hour', axis=1)
corr = fraud_data.corr(method='pearson')
plt.figure(figsize=(25,12))

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, 
            cmap='crest', 
            annot=True,
            fmt='.2f', 
            mask=mask)

#Standardisation
scaler = StandardScaler()
numerical_cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
fraud_data[numerical_cols] = scaler.fit_transform(fraud_data[numerical_cols])

#Feature selection
X = fraud_data.drop(['Class'], axis=1)
y = fraud_data['Class']
selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
# Update the fraud_data DataFrame with the selected features
fraud_data = fraud_data[selected_features]
fraud_data['Class'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(fraud_data.drop(['Class'], axis=1), fraud_data['Class'], test_size=0.2, random_state=42)

# Train a logistic regression model on the training set
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Fit Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate performance on testing set
y_pred = rfc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
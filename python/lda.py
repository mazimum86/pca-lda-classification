# -*- coding: utf-8 -*-
"""
Linear Discriminant Analysis (LDA) with Logistic Regression on the Wine Dataset
Created on Thu Jul 3, 2025
@author: USER
"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset
dataset = pd.read_csv('../data/wine.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply LDA (reduce to 2 linear discriminants)
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Train logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict on test set
y_pred = classifier.predict(X_test)

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Function to visualize decision boundary
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.linspace(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 1000),
        np.linspace(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 1000)
    )
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, Z, alpha=0.25, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for index, label in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == label, 0],
            X_set[y_set == label, 1],
            c=ListedColormap(('red', 'green', 'blue'))(index),
            label=f'Class {label}'
        )

    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualize training set
plot_decision_boundary(X_train, y_train, 'LDA (2D) - Wine Dataset (Training Set)')

# Visualize test set
plot_decision_boundary(X_test, y_test, 'LDA (2D) - Wine Dataset (Test Set)')

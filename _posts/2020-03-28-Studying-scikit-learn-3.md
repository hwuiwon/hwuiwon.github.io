---
layout: post
title: Studying Scikit-learn (3)
subtitle : Regularization
tags: [Machine_Learning, Google_colab, tensorflow, python]
author: Huey Kim
comments : False
---

Regularization are techniques used to reduce the error by fitting a function appropriately on the given training set and avoid overfitting.

## **L1 Regularization (Lasso Regression)**

The constant value is subtracted regardless of the size of the weight. This generally applies in a way that makes the number of unnecessary weights zero. In other words, it is suitable for constructing a model for sparse features because it takes only important weights.

## **L2 Regularization (Ridge Regression)**

For L2, the weight value is used. It means that we can respond to a certain amount of bouncing. Therefore, it is recommended to use it when learning about outliers or noisy data. It is especially good for generalizing linear models.

## **Implementation**

Change the penalty parameter to use L1 or L2.


```
sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
sgd.fit(x_train_scaled, y_train)
sgd.score(x_val_scaled, y_val)
```




    0.978021978021978



There is no significant change in accuracy because there are only small number of data in a set we used.

# **Cross Validation**

This technique is used to evaluate machine learning models on a limited data sample.

The procedure has a single parameter called **k** that refers to the number of blocks that a given data will be split into.<br>These blocks are also called as **folds**.

## **Procedure**

1. Split the training set into k folds.
2. Use the first fold as validation set and remaining folds as training set.
3. Train the model and validate it using validation set.
4. Repeat in turn using the next fold as a validation set.
5. Evaluate the performance k times by using k validation sets and calculate the mean of accuracy to find final performance.

## **Implementation**

We will use **cross_validate()** method and **pipeline** to implement cross validation.

We are using Pipeline to prevent **data leakage during cross validation**.


```
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sgd = SGDClassifier(loss='log', penalty='l2', alpha=0.001, random_state=42)
pipe = make_pipeline(StandardScaler(), sgd)
scores = cross_validate(pipe, x_train_all, y_train_all, cv=10, return_train_score=True)
print(np.mean(scores['test_score']))
```

    0.9694202898550724


**Parameters**

> **alpha**: Constant that multiplies the regularization term.

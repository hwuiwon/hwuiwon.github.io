---
layout: post
title: Predicting Boston House Price
subtitle : First Project
tags: [Machine_Learning, Google_colab, Keras, Python]
author: Huey Kim
comments : False
---

This project will utilize Keras' Boston housing price regression [dataset](https://keras.io/datasets).

[Github Repo](https://github.com/hwuiwon/predict-house-price)

## **Importing Required Libraries**


```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## **Loading Data**

We will load the data by using the method Keras recommends.


```
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

    Using TensorFlow backend.


    Downloading data from https://s3.amazonaws.com/keras-datasets/boston_housing.npz
    57344/57026 [==============================] - 0s 2us/step



```
print(x_train.shape)
print(x_test.shape)
```

    (404, 13)
    (102, 13)


We can see that there are total 506 house data in the set with 13 features.

> **CRIM**: Per capita crime rate by town
<br>**ZN**: Proportion of residential land zoned for lots larger than 25,000 sq.ft.
<br>**INDUS**: Proportion of non-retail business acres per town.
<br>**CHAS**: Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
<br>**NOX**: Nitric oxides concentration (parts per 10 million)
<br>**RM**: Average number of rooms per dwelling
<br>**AGE**: Proportion of owner-occupied units built prior to 1940
<br>**DIS**: Weighted distances to five Boston employment centers
<br>**RAD**: Index of accessibility to radial highways
<br>**TAX**: Full-value property-tax rate per \$10,000
<br>**PTRATIO**: Pupil-teacher ratio by town
<br>**B**: Calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African American descent by town
<br>**LSTAT**: Percentage lower status of the population
<br>**MEDV**: Median value of owner-occupied homes in \$1000

We will convert array to DataFrame to see labels with values.


```
boston = pd.DataFrame(x_train, 
                  index=[i for i in range(x_train.shape[0])], 
                  columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
boston['MEDV'] = y_train

boston.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.23247</td>
      <td>0.0</td>
      <td>8.14</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.142</td>
      <td>91.7</td>
      <td>3.9769</td>
      <td>4.0</td>
      <td>307.0</td>
      <td>21.0</td>
      <td>396.90</td>
      <td>18.72</td>
      <td>15.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02177</td>
      <td>82.5</td>
      <td>2.03</td>
      <td>0.0</td>
      <td>0.415</td>
      <td>7.610</td>
      <td>15.7</td>
      <td>6.2700</td>
      <td>2.0</td>
      <td>348.0</td>
      <td>14.7</td>
      <td>395.38</td>
      <td>3.11</td>
      <td>42.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.89822</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.631</td>
      <td>4.970</td>
      <td>100.0</td>
      <td>1.3325</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>375.52</td>
      <td>3.26</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03961</td>
      <td>0.0</td>
      <td>5.19</td>
      <td>0.0</td>
      <td>0.515</td>
      <td>6.037</td>
      <td>34.5</td>
      <td>5.9853</td>
      <td>5.0</td>
      <td>224.0</td>
      <td>20.2</td>
      <td>396.90</td>
      <td>8.01</td>
      <td>21.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.69311</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.376</td>
      <td>88.4</td>
      <td>2.5671</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>391.43</td>
      <td>14.65</td>
      <td>17.7</td>
    </tr>
  </tbody>
</table>
</div>




```
# Generate descriptive statistics
boston.describe().round(decimals=2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
      <td>404.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.75</td>
      <td>11.48</td>
      <td>11.10</td>
      <td>0.06</td>
      <td>0.56</td>
      <td>6.27</td>
      <td>69.01</td>
      <td>3.74</td>
      <td>9.44</td>
      <td>405.90</td>
      <td>18.48</td>
      <td>354.78</td>
      <td>12.74</td>
      <td>22.40</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.24</td>
      <td>23.77</td>
      <td>6.81</td>
      <td>0.24</td>
      <td>0.12</td>
      <td>0.71</td>
      <td>27.94</td>
      <td>2.03</td>
      <td>8.70</td>
      <td>166.37</td>
      <td>2.20</td>
      <td>94.11</td>
      <td>7.25</td>
      <td>9.21</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.38</td>
      <td>3.56</td>
      <td>2.90</td>
      <td>1.13</td>
      <td>1.00</td>
      <td>188.00</td>
      <td>12.60</td>
      <td>0.32</td>
      <td>1.73</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.08</td>
      <td>0.00</td>
      <td>5.13</td>
      <td>0.00</td>
      <td>0.45</td>
      <td>5.87</td>
      <td>45.48</td>
      <td>2.08</td>
      <td>4.00</td>
      <td>279.00</td>
      <td>17.23</td>
      <td>374.67</td>
      <td>6.89</td>
      <td>16.67</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.27</td>
      <td>0.00</td>
      <td>9.69</td>
      <td>0.00</td>
      <td>0.54</td>
      <td>6.20</td>
      <td>78.50</td>
      <td>3.14</td>
      <td>5.00</td>
      <td>330.00</td>
      <td>19.10</td>
      <td>391.25</td>
      <td>11.40</td>
      <td>20.75</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.67</td>
      <td>12.50</td>
      <td>18.10</td>
      <td>0.00</td>
      <td>0.63</td>
      <td>6.61</td>
      <td>94.10</td>
      <td>5.12</td>
      <td>24.00</td>
      <td>666.00</td>
      <td>20.20</td>
      <td>396.16</td>
      <td>17.09</td>
      <td>24.80</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.98</td>
      <td>100.00</td>
      <td>27.74</td>
      <td>1.00</td>
      <td>0.87</td>
      <td>8.72</td>
      <td>100.00</td>
      <td>10.71</td>
      <td>24.00</td>
      <td>711.00</td>
      <td>22.00</td>
      <td>396.90</td>
      <td>37.97</td>
      <td>50.00</td>
    </tr>
  </tbody>
</table>
</div>



## **Analyzing relationships**

We will now create a heatmap to plot the correlation matrix to see the relationships between the variables.


```
correlation_matrix = boston.corr().round(2)
plt.figure(figsize=(12, 10))
sns.heatmap(data=correlation_matrix, annot=True, annot_kws={"fontsize":10})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fad8645b0f0>




![graph1](/assets/img/posts/p6_graph_1.png)


The correlation coefficient ranges from -1 to 1. If the absolute value is close to 1, it means that there is a strong positive correlation between the two variables and vice versa.

The features of interest are the ones with a high correlation with the target variable 'MEDV'. In this case, we will pick 'RM' and 'LSTAT' as their absolute value is greater than 0.5.


```
g = sns.PairGrid(boston, y_vars=['MEDV'], x_vars=['RM', 'LSTAT'], height=4)
g.map(sns.regplot)
```




    <seaborn.axisgrid.PairGrid at 0x7facdc1ed908>




![graph2](/assets/img/posts/p6_graph_2.png)


We will only include 'RM', 'LSTAT', and 'MEDV' in our data set for accuracy of our model.


```
bostonT = pd.DataFrame(x_test, 
                  index=[i for i in range(x_test.shape[0])], 
                  columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

X_train = pd.DataFrame(np.c_[boston['RM'], boston['LSTAT']], columns = ['RM','LSTAT'])
X_test = pd.DataFrame(np.c_[bostonT['RM'], bostonT['LSTAT']], columns = ['RM','LSTAT'])
```

## **Training the Model**

We will train the model using LinearRegression from scikit-learn.


```
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



We will check the coefficient and intercept of our linear regression model.


```
print(pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficient']))
print(model.intercept_)
```

           Coefficient
    RM        4.866252
    LSTAT    -0.636989
    0.01361359430417508


## **Evaulating**

We will now check the predicted values and the actual values.


```
y_pred = model.predict(X_test)
pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.2</td>
      <td>12.818543</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.8</td>
      <td>18.462167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.0</td>
      <td>22.933857</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>26.924248</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.2</td>
      <td>24.423753</td>
    </tr>
  </tbody>
</table>
</div>



We will plot the y_test vs y_pred. Ideally, it should have been a **straight line**.


```
plt.scatter(y_test, y_pred)
plt.show()
```


![graph3](/assets/img/posts/p6_graph_3.png)


To evaluate the model in detail, we will use the R2-score.


```
from sklearn.metrics import r2_score

y_pred_train = model.predict(X_train)

r2 = r2_score(y_train, y_pred_train)
print("The model performance for training set")
print("--------------------------------------")
print('R2 score is {}'.format(r2))

r2T = r2_score(y_test, y_pred)
print("\nThe model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(r2T))
```

    The model performance for training set
    --------------------------------------
    R2 score is 0.6222162281754451
    
    The model performance for testing set
    --------------------------------------
    R2 score is 0.700683951997074


This implies **70.01%** of variation is explained by the target variable.

We will try to improve the accuracy of a model.

## **Improving the model**

### **Preprocessing the data**

#### **Removing outliers**

We will use a Z-score function defined in scipy library to detect the outliers and remove them.


```
from scipy import stats

z = np.abs(stats.zscore(boston))
threshold = 3
boston_new = boston[(z < 3).all(axis=1)]
```

We removed data that had z score greater than 3.<br>Let's see the results.


```
print(boston.shape)
print(boston_new.shape)
```

    (404, 14)
    (329, 14)


We will now train the model with our modified set.


```
X_train_new = pd.DataFrame(np.c_[boston_new['RM'], boston_new['LSTAT']], columns = ['RM','LSTAT'])
Y_train_new = pd.DataFrame(np.c_[boston_new['MEDV']], columns = ['MEDV'])
```


```
model2 = LinearRegression()
model2.fit(X_train_new, Y_train_new)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



##### **Results: outliers removed**

We will use the R2-score to evaluate the model in detail.


```
y_pred_train_new = model2.predict(X_train_new)
y_pred_test_new = model2.predict(X_test)

r2 = r2_score(Y_train_new, y_pred_train_new)
print("The model performance for training set")
print("--------------------------------------")
print('R2 score is {}'.format(r2))

r2T = r2_score(y_test, y_pred_test_new)
print("\nThe model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(r2T))
```

    The model performance for training set
    --------------------------------------
    R2 score is 0.6901016853316007
    
    The model performance for testing set
    --------------------------------------
    R2 score is 0.6847957414651276


It seems that the accuracy of our training set has improved by 7%, but the overall accuracy is still under 70%.

We will now standardize the data and see if it has any significant effects in our model.

#### **Standardization**

We will standardize the data by using StandardScaler.


```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_new)
X_test_scaled = scaler.transform(X_test)
```

Now we will trian the model with standardized data sets.


```
model3 = LinearRegression()
model3.fit(X_train_scaled, Y_train_new)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



##### **Results: Standarization**

We will use the R2-score to evaluate the model in detail.


```
y_pred_train_new_scaled = model3.predict(X_train_scaled)
y_pred_test_new_scaled = model3.predict(X_test_scaled)

r2 = r2_score(Y_train_new, y_pred_train_new_scaled)
print("The model performance for training set")
print("--------------------------------------")
print('R2 score is {}'.format(r2))

r2T = r2_score(y_test, y_pred_test_new_scaled)
print("\nThe model performance for testing set")
print("--------------------------------------")
print('R2 score is {}'.format(r2T))
```

    The model performance for training set
    --------------------------------------
    R2 score is 0.6901016853316007
    
    The model performance for testing set
    --------------------------------------
    R2 score is 0.684795741465127


## **Conclusion**

There were no significant change in model's accuracy.

There can be various reasons why we are getting low accuracy. I believe this might be happening because there are only a small number of samples that can be used to train the model or there are other features that decide the price of a house that are not in this data set.

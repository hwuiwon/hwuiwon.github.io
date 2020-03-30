---
layout: post
title: Studying Scikit-learn (2)
subtitle : Performance Improvement
tags: [Machine_Learning, Google_colab, Tensorflow, Python]
author: Huey Kim
comments : False
---

## **Improving Performance**

### **Data Preprocessing**

Data collected in real life may not be uniform, unlike data sets in scikit-learn and other machine learning packages that are well processed.<br>If we use those data without preprocessing, we won't be able to yield accurate results.

#### **Problem: Scale Difference**

If the scale of data's characteristics differs a lot with each other, our model may not find  the optimal value of $w$ and b.


```
print(cancer.feature_names[[2, 3]])
plt.boxplot(x_train[:, 2:4])
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
```

    ['mean perimeter' 'mean area']



![graph1](/assets/img/posts/p4_graph_1.png)


By looking at the graph, we can see that the scale of 'mean perimeter' and 'mean area' is different.<br><br>Values of mean perimeter are distributed between **100 ~ 200** while values of mean area are distributed between **200 ~ 2,000**.
<br><br>
What will happen to $w$ if we apply logistic regression function with these values?

We will find out by modifying SingleLayer class that was created previously.


```
class SingleLayer2:

  # Added weight_history & learning_rate
  def __init__(self, learning_rate=0.1):
    self.w = None
    self.b = None
    self.losses = []
    self.w_history = []
    self.lr = learning_rate

  def forpass(self, x):
    z = np.sum(x * self.w) + self.b
    return z

  def backprop(self, x, err):
    w_grad = x * err
    b_grad = 1 * err
    return w_grad, b_grad

  def add_bias(self, x):
    return np.c_[np.ones((x.shape[0], 1)), x]

  def activation(self, z):
    a = 1 / (1 + np.exp(-z))
    return a
  
  # Record weight history
  def fit(self, x, y, epochs=100):
    self.w = np.ones(x.shape[1])
    self.b = 0
    self.w_history.append(self.w.copy())
    for i in range(epochs):
      loss = 0
      indexes = np.random.permutation(np.arange(len(x)))
      for i in indexes:
        z = self.forpass(x[i])
        a = self.activation(z)
        err = -(y[i] - a)
        w_grad, b_grad = self.backprop(x[i], err)
        self.w -= self.lr * w_grad                          # Multiply weight gradient with learning rate 
        self.b -= b_grad
        self.w_history.append(self.w.copy())                # Record weight history
        a = np.clip(a, 1e-10, 1 - 1e-10)

        loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
      self.losses.append(loss/len(y))

  def predict(self, x):
    z = [self.forpass(x_i) for x_i in x]
    return np.array(z) > 0

  def score(self, x, y):
    return np.mean(self.predict(x) == y)
```

We made following changes to original SingleLayer:

1. Added weight_history & learning_rate when it gets initialized.
2. Record weight history and multiply learning weight with weight gradient when changing the weight.

<br><br>
Now lets test and see the results.


```
layer1 = SingleLayer2()
layer1.fit(x_train, y_train)
layer1.score(x_val, y_val)
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:24: RuntimeWarning: overflow encountered in exp





    0.9120879120879121



Before we adjust the scale of data's characteristics, we got **91.2%** accuracy.
<br><br>
Now we will see how our weight changed over each epoch.<br>Final value of $w$ will be shown as a red dot.


```
w2 = []
w3 = []
for w in layer1.w_history:
  w2.append(w[2])
  w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w[2]')
plt.ylabel('w[3]')
plt.show()
```


![graph2](/assets/img/posts/p4_graph_2.png)


By looking at the graph, we can see the value of w3 **fluctuates widely**.<br>This is because as the gradient for w3 is large, the weight fluctuates greatly along the w3 axis.
<br>
We can improve our results by using **standarization**.

#### **Solution: Standarization**

Formula: z = (x - mu) / s, where x is value, mu is mean, and s is standard deviation.

<br>
We will calculate mean and standard deviation by using numpy.


```
import numpy as np

train_mean = np.mean(x_train, axis=0)
train_std = np.std(x_train, axis=0)
x_train_scaled = (x_train - train_mean) / train_std
```

Now we will trian the model with standatized data sets using SingleLayer2.


```
layer2 = SingleLayer2()
layer2.fit(x_train_scaled, y_train)
w2 = []
w3 = []
for w in layer2.w_history:
  w2.append(w[2])
  w3.append(w[3])
plt.plot(w2, w3)
plt.plot(w2[-1], w3[-1], 'ro')
plt.xlabel('w[2]')
plt.ylabel('Standarized w[3]')
plt.show()
```


![graph3](/assets/img/posts/p4_graph_3.png)


We will check whether this this model yields better accuracy than previous model without standarization.


```
x_val_scaled = (x_val - train_mean) / train_std
layer2.score(x_val_scaled, y_val)
```




    0.967032967032967



The accuracy of a model **improved** by approximately 5.5%. (0.912 -> 0.967)

It is important to **scale training set and validation set with same ratio**.

### **Overfitting & Underfitting**

#### **Definition**

**Overfitting**: The model performs well with training set, but doesn't perform well with validation set.<br>Ex) 99% accuracy with training set but 80% accuracy with validation set
<br><br>
**Underfitting**: The performace between training set and validation set is similar but it's overall accuracy is low.
<br><br>
We also say overfitted model has **high variance** and unferfitted model is **highly biased**.

#### **Main Cause**
<br>
**Overfitting**

> Samples of sufficiently diverse patterns were not included in the training set.<br>Overfitted model can be improved by collecting more training data.

> If it's realistically hard to collect more data, we can **regularize** the weight $w$.

**Underfitting**

> Model isn't complex enough to analyze the data set and return the right result.<br>Underfitted model can be improved by using more complex models.

![graph4](/assets/img/posts/p4_graph_4.png)

Where x-axis is **epochs** and y-axis is **accuracy**.

#### **Bias-variance Tradeoff**

Models with a lower bias in parameter estimation have a higher variance of the parameter estimates across samples, and vice versa.
<br><br>
We need to select an appropriate point to prevent high bias and variance.<br>To record losses in validation, we will make edits in SingleLayer2 class.


```
class SingleLayer3:

  # Added validation losses
  def __init__(self, learning_rate=0.1):
    self.w = None
    self.b = None
    self.losses = []
    self.val_losses = []
    self.w_history = []
    self.lr = learning_rate

  def forpass(self, x):
    z = np.sum(x * self.w) + self.b
    return z

  def backprop(self, x, err):
    w_grad = x * err
    b_grad = 1 * err
    return w_grad, b_grad

  def add_bias(self, x):
    return np.c_[np.ones((x.shape[0], 1)), x]

  def activation(self, z):
    a = 1 / (1 + np.exp(-z))
    return a
  
  
  def fit(self, x, y, epochs=100, x_val=None, y_val=None):
    self.w = np.ones(x.shape[1])
    self.b = 0
    self.w_history.append(self.w.copy())
    for i in range(epochs):
      loss = 0
      indexes = np.random.permutation(np.arange(len(x)))
      for i in indexes:
        z = self.forpass(x[i])
        a = self.activation(z)
        err = -(y[i] - a)
        w_grad, b_grad = self.backprop(x[i], err)
        self.w -= self.lr * w_grad
        self.b -= b_grad
        self.w_history.append(self.w.copy())
        a = np.clip(a, 1e-10, 1 - 1e-10)
        loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))

      self.losses.append(loss/len(y))
      self.update_val_loss(x_val, y_val)          # Update validation loss for each epoch

  def predict(self, x):
    z = [self.forpass(x_i) for x_i in x]
    return np.array(z) > 0

  def score(self, x, y):
    return np.mean(self.predict(x) == y)

  # Method that updates validation losses
  def update_val_loss(self, x_val, y_val):
    if x_val is None:
      return
    val_loss = 0
    for i in range(len(x_val)):
      z = self.forpass(x_val[i])
      a = self.activation(z)
      a = np.clip(a, 1e-10, 1-1e-10)
      val_loss += -(y_val[i] * np.log(a) + (1 - y_val[i]) * np.log(1 - a))
    self.val_losses.append(val_loss / len(y_val))
```

Let's train our model with an modified model.


```
layer3 = SingleLayer3()
layer3.fit(x_train_scaled, y_train, x_val=x_val_scaled, y_val=y_val)
```

Now we will draw a graph to see validation losses.


```
plt.ylim(0, 0.3)
plt.plot(layer3.losses)
plt.plot(layer3.val_losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()
```


![graph5](/assets/img/posts/p4_graph_5.png)


We can see that validation loss gets higher than training set after 20 epochs.<br><br>At this point, as we go through more epochs to train our model, $w$ will **fit better with training set**, but less with validation set.
<br><br>
So after going through 20 epochs, there is no need to train anymore.<br>We call this technique '**early stopping**'.


```
layer4 = SingleLayer3()
layer4.fit(x_train_scaled, y_train, epochs=20)
layer4.score(x_val_scaled, y_val)
```




    0.978021978021978



The accuracy of a model **improved** by approximately 1.1%. (0.967 -> 0.978)

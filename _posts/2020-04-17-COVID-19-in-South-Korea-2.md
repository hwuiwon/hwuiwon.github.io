---
layout: post
title: COVID-19 in South Korea (2)
subtitle : Analyzing Time and Search Trend
tags: [Google_colab, Python, Pandas, Seaborn, Data Analysis]
author: Huey Kim
comments : False
---


# **covid-19-in-South-Korea**

*Author*: Huey Kim [Github](https://github.com/hwuiwon)

We will use [Data Science for COVID-19 dataset](https://www.kaggle.com/kimjihoo/coronavirusdataset) provided by DS4C at Kaggle.

## **Introduction**

### **List of Files**

*Encoding*: UTF-8

*   **SearchTrend.csv** (Trend data of the keywords searched in NAVER which is one of the largest portals) ✔️
*   **Time.csv** (Time series data of COVID-19 status) ✔️

## **By Time**

### **Assumption**

Graph of COVID-19 confirmed cases will follow SIR epidemic model.

![SIR Model](https://drive.google.com/uc?id=1rjMH9cdoP_V9Kt2OGCfQHP5NJoBxSPA9)

where

> **S(t)** are those susceptible but not yet infected with the disease<br>
> **I(t)** is the number of infectious individuals<br>
> **R(t)** are those individuals who have recovered from the disease and now have immunity to it.

### **Visualization**


```
# Importing data: Time
time = pd.read_csv(os.path.join(DIR_PATH, 'Time.csv'))
time.head()
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
      <th>date</th>
      <th>time</th>
      <th>test</th>
      <th>negative</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-20</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-21</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-22</td>
      <td>16</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-23</td>
      <td>16</td>
      <td>22</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-24</td>
      <td>16</td>
      <td>27</td>
      <td>25</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Number of tests conducted
ax1.title.set_text('Total COVID-19 Tests Conducted')
sns.lineplot(data=time, x='date', y='test', label='total', ax=ax1)
sns.lineplot(data=time, x='date', y='confirmed', color='red', label='positive', ax=ax1)
sns.lineplot(data=time, x='date', y='negative', color='green', label='negative', ax=ax1)

ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
ax1.set(ylabel='count')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Positive & Released & Deceased cases
ax2.title.set_text('Patient Count')
sns.lineplot(data=time, x='date', y='confirmed', label='positive', ax=ax2)
sns.lineplot(data=time, x='date', y='released', label='released', ax=ax2)
sns.lineplot(data=time, x='date', y='deceased', label='deceased', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
ax2.set(ylabel='count')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Draw vertical line in patient count graph
ax2.axvline('2020-03-10', 0, 10000, color='red', linestyle='dotted')

plt.show()
```


![graph1](/assets/img/posts/p9_graph_1.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))
time['p_growth_rate'] = time['confirmed'].pct_change()
time['n_growth_rate'] = time['negative'].pct_change()

# Growth rate of positive cases
ax1.set_title('Positive Case Growth Rate')
sns.lineplot(data=time, x='date', y='p_growth_rate', ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Growth rate of negative cases
ax2.set_title('Negative Case Growth Rate')
sns.lineplot(data=time, x='date', y='n_growth_rate', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.show()
```


![graph2](/assets/img/posts/p9_graph_2.png)



```
# Proportion by total population
time_f = time.tail(1)
time_latestT = time_f.test.values[0]
time_latestP = time_f.confirmed.values[0]
time_latestN = time_f.negative.values[0]
pop_total = int(pop_dist.iat[0, 1].replace(',', ''))

print('Percentage of people tested out of total population: {0}%\n'.format(round(time_latestT / pop_total * 100, 2)) + 
      'Percentage of positive cases out of people tested: {0}%\n'.format(round(time_latestP / time_latestT * 100, 2)) + 
      'Percentage of negative cases out of people tested: {0}%'.format(round(time_latestN / time_latestT * 100, 2)))
```

    Percentage of people tested out of total population: 0.92%
    Percentage of positive cases out of people tested: 2.16%
    Percentage of negative cases out of people tested: 93.51%


### **Analysis & Conclusion**

1. Total number of conducted tests and negative results are increasing linearly while the rate of positive results is slowly decreasing unlike its exponential growth in the first.
2. Since 2020-03-10, rate of positive results is decreasing and rate of released patients is increasing rapidly.
3. From spikes in the positive case growth rate, we can infer some event has happened just before, causing collective infection.

A Graph of time vs infection cases follows the SIR epidemic model.

## **By Search Trend**

### **Assumption**

The search keywords related to COVID-19 would have been searched the most when the growth rate of positive cases is at its maximum.

### **Visualization**


```
# Importing data: Search Trend
searchtrend = pd.read_csv(os.path.join(DIR_PATH, 'SearchTrend.csv'))
searchtrend.head()
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
      <th>date</th>
      <th>cold</th>
      <th>flu</th>
      <th>pneumonia</th>
      <th>coronavirus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-01-01</td>
      <td>0.11663</td>
      <td>0.05590</td>
      <td>0.15726</td>
      <td>0.00736</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-01-02</td>
      <td>0.13372</td>
      <td>0.17135</td>
      <td>0.20826</td>
      <td>0.00890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-01-03</td>
      <td>0.14917</td>
      <td>0.22317</td>
      <td>0.19326</td>
      <td>0.00845</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-01-04</td>
      <td>0.17463</td>
      <td>0.18626</td>
      <td>0.29008</td>
      <td>0.01145</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-01-05</td>
      <td>0.17226</td>
      <td>0.15072</td>
      <td>0.24562</td>
      <td>0.01381</td>
    </tr>
  </tbody>
</table>
</div>




```
searchTrend_2020 = searchtrend.iloc[1461:, :]
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Search proportion of keywords related to COVID-19
ax1.title.set_text('Keyword Search Trend')
for keyword in searchTrend_2020.iloc[:, 1:].columns:
    sns.lineplot(data=searchTrend_2020, x='date', y=keyword, label=keyword, ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
ax1.set(ylabel='percentage')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Search proportion of keywords related to COVID-19 except coronavirus
ax2.title.set_text('Keyword Search Trend (excluding "coronavirus")')
sns.lineplot(data=searchTrend_2020, x='date', y='cold', label='cold', ax=ax2)
sns.lineplot(data=searchTrend_2020, x='date', y='flu', label='flu', ax=ax2)
sns.lineplot(data=searchTrend_2020, x='date', y='pneumonia', label='pneumonia', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
ax2.set(ylabel='percentage')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.show()
```


![graph3](/assets/img/posts/p9_graph_3.png)


### **Analysis & Conclusion**

1. Between Jan 8, 2020 and Jan 11, 2020, there were sudden spike of keyword search related to coronavirus.
2. After Jan 11, 2020, people searched less for cold, flu, and pneumonia and percentage of coronavirus searched through NAVER (top online portal in KR) increased rapidly starting from Jan 9, 2020, maintaining its search frequency over 50% most of the time until March 4, 2020.
3. We can assume people first became aware of corona virus at Jan 8, 2020 and people's awareness became high when the first COVID-19 patient was spotted in South Korea at Jan 20, 2020.
4. We can see that search trend of coronavirus has once again spiked in Feb 18, 2020, as number of confirmed cases suddenly grew exponentially.
5. However, corona virus is receiving less attention since new confirmed cases of COVID-19 have decreased significantly compared to previous days. (new cases < 30)

People tend to search keywords related to COVID-19 when it was first brought to spotlight, when first death case happened, and when there was significant increase in growth rate of confirmed cases.

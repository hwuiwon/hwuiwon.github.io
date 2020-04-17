---
layout: post
title: COVID-19 in South Korea (1)
subtitle : Analyzing Gender, Age, and Location
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

*   **Region.csv** (Location and statistical data of the regions) ✔️
*   **TimeAge.csv** (Time series data of COVID-19 status in terms of the age) ✔️
*   **TimeGender.csv** (Time series data of COVID-19 status in terms of gender) ✔️
*   **TimeProvince.csv** (Time series data of COVID-19 status in terms of the Province) ✔️

### **Setting Environment**


```
DIR_PATH = '/content/drive/My Drive/Colab Notebooks/data/kr-corona-dataset/'
```


```
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

import os;
import numpy as np;
import pandas as pd;
import seaborn as sns;
import folium;
import matplotlib.pyplot as plt;
import matplotlib.ticker as ticker

sns.set_style('darkgrid')
```

### **External Datasets**

**Population Distribution Data**

* from [Ministry of the Interior and Safety](http://27.101.213.4/)

File Name: *PopulationDistribution.csv* (as of March)

Use this code to convert file with encoding **EUC-KR** to **UTF-8**.
```
iconv -f euc-kr -t utf-8 old.csv > new.csv
```


```
# Importing data: PopulationDistribution
pop_dist = pd.read_csv(os.path.join(DIR_PATH, 'PopulationDistribution.csv'))
pop_dist = pop_dist.iloc[:, np.r_[0, 1, 3:12, 14:15, 27]]
pop_dist.columns = ['location', 'total', '0s', '10s', '20s', '30s', '40s', '50s',
                   '60s', '70s', '80s', 'male_total', 'female_total']
pop_dist.head()
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
      <th>location</th>
      <th>total</th>
      <th>0s</th>
      <th>10s</th>
      <th>20s</th>
      <th>30s</th>
      <th>40s</th>
      <th>50s</th>
      <th>60s</th>
      <th>70s</th>
      <th>80s</th>
      <th>male_total</th>
      <th>female_total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>전국  (0000000000)</td>
      <td>51,843,195</td>
      <td>4,119,475</td>
      <td>4,902,009</td>
      <td>6,799,238</td>
      <td>7,006,749</td>
      <td>8,375,429</td>
      <td>8,662,400</td>
      <td>6,426,006</td>
      <td>3,623,899</td>
      <td>1,667,665</td>
      <td>25,858,743</td>
      <td>25,984,452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울특별시  (1100000000)</td>
      <td>9,733,655</td>
      <td>655,134</td>
      <td>810,349</td>
      <td>1,461,212</td>
      <td>1,501,662</td>
      <td>1,558,128</td>
      <td>1,537,114</td>
      <td>1,205,699</td>
      <td>691,028</td>
      <td>266,097</td>
      <td>4,742,217</td>
      <td>4,991,438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>부산광역시  (2600000000)</td>
      <td>3,409,932</td>
      <td>240,857</td>
      <td>279,909</td>
      <td>435,280</td>
      <td>430,494</td>
      <td>516,889</td>
      <td>579,584</td>
      <td>513,485</td>
      <td>286,808</td>
      <td>110,254</td>
      <td>1,672,252</td>
      <td>1,737,680</td>
    </tr>
    <tr>
      <th>3</th>
      <td>대구광역시  (2700000000)</td>
      <td>2,433,568</td>
      <td>185,945</td>
      <td>233,311</td>
      <td>322,594</td>
      <td>302,617</td>
      <td>390,628</td>
      <td>428,955</td>
      <td>313,213</td>
      <td>171,945</td>
      <td>74,745</td>
      <td>1,202,273</td>
      <td>1,231,295</td>
    </tr>
    <tr>
      <th>4</th>
      <td>인천광역시  (2800000000)</td>
      <td>2,952,689</td>
      <td>238,763</td>
      <td>279,780</td>
      <td>405,826</td>
      <td>416,119</td>
      <td>489,213</td>
      <td>515,257</td>
      <td>347,493</td>
      <td>172,868</td>
      <td>74,768</td>
      <td>1,479,839</td>
      <td>1,472,850</td>
    </tr>
  </tbody>
</table>
</div>



**Province Geolocation Data**
* from [GEOSERVICE](http://www.gisdeveloper.co.kr/?p=2332)

File Name: *province_geo.json*

### **Methods for Better Visualization**


```
# Tag value on bars
def show_values_on_bars(axs, h_v="v", space=0.4, modh=0, modv=0):
    def _show_on_single_plot(ax):
        if h_v == 'v':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + float(modv)
                value = int(p.get_height())
                ax.text(_x, _y, value, ha='center') 
        elif h_v == 'h':
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - float(modh)
                value = int(p.get_width())
                ax.text(_x, _y, value, ha='left')

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
```

## **By Gender**

### **Assumption**

There will be no correlation between gender and COVID-19 infection.

### **Visualization**


```
# Importing data: Gender
gender = pd.read_csv(os.path.join(DIR_PATH, 'TimeGender.csv'))
gender.head(2)
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
      <th>sex</th>
      <th>confirmed</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>male</td>
      <td>1591</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>female</td>
      <td>2621</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17, 7), gridspec_kw={'width_ratios': [1, 2]})
sns.set_palette(['#66b3ff','#ff9999'])

# Donut plot of confirmed cases by gender
ax1.title.set_text('Confirmed Cases ({0})'.format(gender.iloc[-1, 0]))
ax1.pie(gender.confirmed[-2:], labels=['male', 'female'], autopct='%.1f%%',
        startangle=90, pctdistance=0.85)
ax1.add_artist(plt.Circle((0, 0), 0.7, fc='white'))

# Change in time of confirmed cases
ax2.title.set_text('Confirmed Cases by Gender')
sns.lineplot(data=gender, x='date', y='confirmed', hue='sex', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
```


![graph1](/assets/img/posts/p8_graph_1.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Growth rate of confirmed cases (Index - male: even, female: odd)
ax1.title.set_text('Growth Rate of Confirmed Cases by Gender')
gender['growth_rate'] = gender.groupby('sex')[['confirmed']].pct_change()
sns.lineplot(data=gender, x='date', y='growth_rate', hue='sex', ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Decease rate of confirmed cases
ax2.title.set_text('Decease Rate of Confirmed Cases by Gender')
# Limiting y axis range to reduce fluctuations in graph
ax2.set(ylim=(-0.05, 0.5))
gender['decease_rate'] = gender.groupby('sex')[['deceased']].pct_change()
sns.lineplot(data=gender, x='date', y='decease_rate', hue='sex', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.xticks(rotation=45, ha='right')

plt.show()
```


![graph2](/assets/img/posts/p8_graph_2.png)


### **Analysis & Conclusion**

1. More females are infected than males.
2. The growth rate of confirmed cases is similar regardless of genders.
3. Both genders show a similar trend of decease rate.

Gender **isn't** a significant factor that influences the infection rate. (No correlation)

The reason why more females were infected than men might be due to geological reasons or different lifestyles.

## **By Age**

### **Assumption**

Age group 10s and 20s will be most infected as they are more active and study in a crowded place such as academy or school.<br>As people age, they will be more susceptible of getting infected as their immune system weakens.

### **Visualization**


```
# Importing data: Age
age = pd.read_csv(os.path.join(DIR_PATH, 'TimeAge.csv'))
print('Unique items: {0}'.format(len(age['age'].unique())))
age.head(9)
```

    Unique items: 9





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
      <th>age</th>
      <th>confirmed</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>0s</td>
      <td>32</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>10s</td>
      <td>169</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>20s</td>
      <td>1235</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>30s</td>
      <td>506</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>40s</td>
      <td>633</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>50s</td>
      <td>834</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>60s</td>
      <td>530</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>70s</td>
      <td>192</td>
      <td>6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-03-02</td>
      <td>0</td>
      <td>80s</td>
      <td>81</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```
sns.set_palette('deep')
pop_dist_age = pop_dist.iloc[0, 2:11].str.replace(',', '')

# Population distribution by age
plt.figure(figsize=(7, 7))
plt.title('Age Distribution in South Korea')
plt.pie(pop_dist_age, labels=pop_dist_age.index, 
        autopct='%.1f%%', startangle=90, pctdistance=0.85)
plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))
plt.show()
```


![graph3](/assets/img/posts/p8_graph_3.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Confirmed cases by age
ax1.title.set_text('Confirmed Cases of COVID-19')
sns.barplot(data=age[-9:], x='age', y='confirmed', ax=ax1)

# Create new column of total people in that age group
pop_dist_age = pop_dist.iloc[0, 2:11].str.replace(',', '')
age['age_total'] = np.tile(pop_dist_age, len(age) // len(pop_dist_age) + 1)[:len(age)]

# Create proportion column
age['prop_total'] = age['confirmed'] / age['age_total'].astype(float)

# Proportion of confirmed cases by age to total people in age group
ax2.title.set_text('Confirmed Cases of COVID-19 (Out of total age group)')
sns.barplot(data=age[-9:], x='age', y='prop_total', ax=ax2)

plt.show()
```


![graph4](/assets/img/posts/p8_graph_4.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Confirmed cases by age
ax1.title.set_text('Confirmed Cases by Age')
sns.lineplot(data=age, x='date', y='confirmed', hue='age', ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Deceased cases by age
ax2.title.set_text('Deceased Cases of Confirmed Cases by Age')
sns.lineplot(data=age, x='date', y='deceased', hue='age', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.xticks(rotation=45, ha='right')

plt.show()
```


![graph5](/assets/img/posts/p8_graph_5.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Growth rate of confirmed cases
ax1.title.set_text('Growth Rate of Confirmed Cases by Age')
age['growth_rate'] = age.groupby('age')[['confirmed']].pct_change()
sns.lineplot(data=age, x='date', y='growth_rate', hue='age', ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Decease rate of confirmed cases
ax2.title.set_text('Decease Rate of Confirmed Cases by Age')
age['decease_rate'] = age.groupby('age')[['deceased']].pct_change()
sns.lineplot(data=age, x='date', y='decease_rate', hue='age', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.xticks(rotation=45, ha='right')

plt.show()
```


![graph6](/assets/img/posts/p8_graph_6.png)


### **Analysis & Conclusion**

1. Although 20s only take up 13.2% of all populations in South Korea, they are the most infected and has the highest infection rate among all other age groups.
2. Older people are more prone to get COVID-19 and are more likely to get deceased.
3. Trend of growth and decease rate seems similar.

Age seems to be a significant feature that influences infection rate.

High infection rate of age group 20s can might be explained by their social activeness as young people tend to move around places more than older people. As people get older, their immunity tends to drop, which might be the reason why there are more cases of infections as people age.

## **By Location**

### **Assumption**

Seoul and metropolitan cities would have relatively high infection cases due to their floating population and high population density.

### **Visualization**

#### **TimeProvince.csv**

Time series data of COVID-19 status in terms of the Province


```
# Importing data: Location
location = pd.read_csv(os.path.join(DIR_PATH, 'TimeProvince.csv'))
prov_num = len(location['province'].unique())
print(f'There are {prov_num} provinces in this dataset')

# Latest data of confirmed cases by province
loc_latest = location.iloc[-prov_num:]
loc_latest = loc_latest.sort_values('confirmed', ascending=False).reset_index(
                        drop=True).drop('time', axis=1)
loc_latest
```

    There are 17 provinces in this dataset





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
      <th>province</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-04-07</td>
      <td>Daegu</td>
      <td>6794</td>
      <td>4918</td>
      <td>134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-04-07</td>
      <td>Gyeongsangbuk-do</td>
      <td>1317</td>
      <td>934</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-07</td>
      <td>Gyeonggi-do</td>
      <td>590</td>
      <td>226</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-07</td>
      <td>Seoul</td>
      <td>567</td>
      <td>164</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-07</td>
      <td>Chungcheongnam-do</td>
      <td>137</td>
      <td>104</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-04-07</td>
      <td>Busan</td>
      <td>123</td>
      <td>91</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020-04-07</td>
      <td>Gyeongsangnam-do</td>
      <td>112</td>
      <td>80</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-04-07</td>
      <td>Incheon</td>
      <td>80</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-04-07</td>
      <td>Gangwon-do</td>
      <td>47</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-04-07</td>
      <td>Sejong</td>
      <td>46</td>
      <td>17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-04-07</td>
      <td>Chungcheongbuk-do</td>
      <td>45</td>
      <td>29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2020-04-07</td>
      <td>Ulsan</td>
      <td>40</td>
      <td>28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2020-04-07</td>
      <td>Daejeon</td>
      <td>39</td>
      <td>20</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2020-04-07</td>
      <td>Gwangju</td>
      <td>27</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2020-04-07</td>
      <td>Jeollabuk-do</td>
      <td>16</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2020-04-07</td>
      <td>Jeollanam-do</td>
      <td>15</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2020-04-07</td>
      <td>Jeju-do</td>
      <td>12</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```
# Latest number of confirmed & released & deceased people
fig, ax1 = plt.subplots(figsize=(15, 7))
ax1.title.set_text('COVID-19 Patients by Province')
sns.set_color_codes("pastel")
sns.barplot(data=loc_latest, x='confirmed', y='province',  label='confirmed',
            color='b', ci=None, estimator=sum)
sns.barplot(data=loc_latest, x='released', y='province', label='released',
            color='r', ci=None, estimator=sum)
sns.barplot(data=loc_latest, x='deceased', y='province', label='deceased',
            color='g', ci=None, estimator=sum)
ax1.legend(loc='lower right', frameon=True)
fig.show()
```


![graph7](/assets/img/posts/p8_graph_7.png)



```
# Confirmed cases in each province (accumulated)
rows = int(prov_num / 2 + 1)
fig, ax = plt.subplots(rows, 2, figsize=(20, 6 * rows))
fig.subplots_adjust(hspace=.3)

for i, province in enumerate(loc_latest['province']):
    r, c = int(i / 2), i % 2
    sns.lineplot(data=location[location['province'] == province],
                 x='date', y='confirmed', ax=ax[r, c])
    ax[r, c].set_title(f'Confirmed Cases in {province}')
    ax[r, c].xaxis.set_major_locator(ticker.MultipleLocator(base=6))
    plt.setp(ax[r, c].xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.delaxes(ax[rows - 1][rows * 2 - prov_num])
fig.show()
```


![graph8](/assets/img/posts/p8_graph_8.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
location['growth_rate'] = location.groupby('province')[['confirmed']].pct_change()

# Growth rate of confirmed cases in Daegu
ax1.set_title('Growth rate of confirmed cases (Daegu)')
sns.lineplot(data=location[location['province'] == 'Daegu'], x='date', y='growth_rate', ax=ax1)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Growth Rate of confirmed cases in Gyeonggi-do
ax2.set_title('Growth rate of confirmed cases (Gyeonggi-do)')
sns.lineplot(data=location[location['province'] == 'Gyeonggi-do'], x='date', y='growth_rate', ax=ax2)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')

fig.show()
```


![graph9](/assets/img/posts/p8_graph_9.png)



```
# Proportion out of total confirmed cases by province
loc_latest['proportion'] = round(loc_latest['confirmed'] / sum(loc_latest['confirmed']) * 100, 2)

# Combine provinces that consists less than 2% of total cases
loc_latest.loc['17',:] = loc_latest.iloc[4:, :].sum()
loc_latest.loc['17',['date', 'province']] = ['2020-03-30', 'Others']

sns.set_palette('deep')
loc_latest_w_etc = loc_latest.iloc[[0, 1, 2, 3, 17], [1, 5]]

# COVID-19 distribution by province
plt.figure(figsize=(7, 7))
plt.title('COVID-19 Distribution by Province')
plt.pie(loc_latest_w_etc['proportion'], labels=loc_latest_w_etc['province'], 
        autopct='%.1f%%', startangle=90, pctdistance=0.85)
plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))
plt.show()
```


![graph10](/assets/img/posts/p8_graph_10.png)


#### **Region.csv**

Location and statistical data of the regions


```
# Importing data: Region
region = pd.read_csv(os.path.join(DIR_PATH, 'Region.csv'))
region = region.drop('nursing_home_count', axis=1)
# region = region.drop(['latitude', 'longitude', 'nursing_home_count'], axis=1)
# Drop column with same value and sort by academy_ratio
region_overview = region[region['province'] == region['city']].drop('city',
                  axis=1).drop(243).sort_values('academy_ratio', 
                  ascending=False).reset_index(drop=True)
region_overview.head()
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
      <th>code</th>
      <th>province</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>elementary_school_count</th>
      <th>kindergarten_count</th>
      <th>university_count</th>
      <th>academy_ratio</th>
      <th>elderly_population_ratio</th>
      <th>elderly_alone_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13000</td>
      <td>Gwangju</td>
      <td>35.160467</td>
      <td>126.851392</td>
      <td>155</td>
      <td>312</td>
      <td>17</td>
      <td>2.38</td>
      <td>13.57</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16000</td>
      <td>Ulsan</td>
      <td>35.539797</td>
      <td>129.311538</td>
      <td>119</td>
      <td>200</td>
      <td>4</td>
      <td>2.21</td>
      <td>11.76</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50000</td>
      <td>Jeollabuk-do</td>
      <td>35.820308</td>
      <td>127.108791</td>
      <td>419</td>
      <td>519</td>
      <td>19</td>
      <td>2.12</td>
      <td>20.60</td>
      <td>10.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17000</td>
      <td>Sejong</td>
      <td>36.480132</td>
      <td>127.289021</td>
      <td>48</td>
      <td>60</td>
      <td>3</td>
      <td>1.78</td>
      <td>9.48</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61000</td>
      <td>Gyeongsangnam-do</td>
      <td>35.238294</td>
      <td>128.692397</td>
      <td>501</td>
      <td>686</td>
      <td>21</td>
      <td>1.78</td>
      <td>16.51</td>
      <td>9.1</td>
    </tr>
  </tbody>
</table>
</div>




```
# Add latitude and longtitude
loc_latest = loc_latest.merge(
    region_overview[['province', 'latitude','longitude']],
    on = 'province')
loc_latest['latitude'] = loc_latest['latitude'].astype(float)
loc_latest['longitude'] = loc_latest['longitude'].astype(float)
loc_latest.head()
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
      <th>province</th>
      <th>confirmed</th>
      <th>released</th>
      <th>deceased</th>
      <th>proportion</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-04-07</td>
      <td>Daegu</td>
      <td>6794.0</td>
      <td>4918.0</td>
      <td>134.0</td>
      <td>67.89</td>
      <td>35.872150</td>
      <td>128.601783</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-04-07</td>
      <td>Gyeongsangbuk-do</td>
      <td>1317.0</td>
      <td>934.0</td>
      <td>46.0</td>
      <td>13.16</td>
      <td>36.576032</td>
      <td>128.505599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-07</td>
      <td>Gyeonggi-do</td>
      <td>590.0</td>
      <td>226.0</td>
      <td>7.0</td>
      <td>5.90</td>
      <td>37.275119</td>
      <td>127.009466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-07</td>
      <td>Seoul</td>
      <td>567.0</td>
      <td>164.0</td>
      <td>0.0</td>
      <td>5.67</td>
      <td>37.566953</td>
      <td>126.977977</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-07</td>
      <td>Chungcheongnam-do</td>
      <td>137.0</td>
      <td>104.0</td>
      <td>0.0</td>
      <td>1.37</td>
      <td>36.658976</td>
      <td>126.673318</td>
    </tr>
  </tbody>
</table>
</div>




```
# COVID-19 infection distribution
map_southKR = folium.Map(location=[35.9, 128], tiles="cartodbpositron",
                         zoom_start=7, max_zoom=9, min_zoom=5)
folium.Choropleth(geo_data=os.path.join(DIR_PATH, 'province_geo.json'),
                  fill_color='#ffff66', line_opacity=0.5, fill_opacity=0.3).add_to(map_southKR)

for i in range(0, len(loc_latest)):
    folium.Circle(
        location=[loc_latest.iloc[i]['latitude'], loc_latest.iloc[i]['longitude']],
        tooltip="<h5 style='text-align:center;font-weight: bold'>" + 
                loc_latest.iloc[i]['province'] + "</h5><hr style='margin:10px;'>" +
                "<ul style='align-item:left;padding-left:20px;padding-right:20px'>" +
                "<li>Confirmed: " + str(loc_latest.iloc[i]['confirmed']) + "</li>" +
                "<li>Deaths: " + str(loc_latest.iloc[i]['deceased']) + "</li>" +
                "<li>Mortality Rate: " + str(round(loc_latest.iloc[i]['deceased'] /
                                                   (loc_latest.iloc[i]['confirmed'] + .000001) * 100, 2)) + 
                "%</li></ul>",
        radius=int((np.log(loc_latest.iloc[i]['confirmed'])))*5000,
        color='#ff3333',
        fill_color='#ff0000',
        fill=True).add_to(map_southKR)

map_southKR
```


```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 5))

# Academy ratio of each province
ax1.title.set_text('Academy Ratio of Each Province')
sns.barplot(data=region_overview, x='province', y='academy_ratio', ax=ax1)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

region_overview = region_overview.sort_values('elderly_population_ratio', 
                                              ascending=False).reset_index(drop=True)

# Elderly population ratio of each province
ax2.title.set_text('Elderly Population Ratio of Each Province')
sns.barplot(data=region_overview, x='province', y='elderly_population_ratio', ax=ax2)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.show()
```


![graph11](/assets/img/posts/p8_graph_11.png)



```
# DataFrame only with province and population value
pop_dist_prov = pop_dist.copy(deep=True)
pop_dist_prov['total'] = pop_dist_prov['total'].str.replace(',', '').astype(int)
pop_dist_prov = pop_dist_prov.sort_values('total', ascending=False).reset_index(
    drop=True).drop(pop_dist_prov.columns[2:13], axis=1)
by_i_case = pop_dist_prov.loc[[7, 6, 1, 2, 8, 3, 4, 5, 17, 11, 15], :]
by_i_case['location'] = ['Daegu','Gyeongsangbuk-do','Gyeonggi-do','Seoul',
                 'Chungcheongnam-do','Busan','Gyeongsangnam-do','Incheon',
                 'Sejong','Chungcheongbuk-do','Ulsan']

# Province population ordered by infection cases
plt.figure(figsize=(10, 5))
plt.title('Province Population (Order by infection cases)')
sns.barplot(data=by_i_case, x='location', y='total')
plt.xticks(rotation=30, ha='right')
plt.show()
```


![graph12](/assets/img/posts/p8_graph_12.png)


### **Analysis & Conclusion**

1. Less than 100 people were infected for the first month but infection case has grown exponentially for the following months.
2. Top 3 provinces with high COVID-19 infection take up 90% of total cases.
3. It seems COVID-19 has gone to a lull (Slope is decreasing) in provinces except Gyeonggi-do, Seoul, and Incheon.
4. The infection growth rate in Gyeonggi-do shows several spikes, which means there were a sudden increase in infection cases that might have been caused by collective infection due to an event or work.
5. Academy ratio and elderly population ratio of a province seem to have no correlation with infection cases in particular province.
6. There tends to be more confirmed cases of COVID-19 in provinces with high popultaion except Daegu and Gyeongsangbuk-do.

Location seems to be a significant feature that influences infection rate.

However, it seems location is not the major factor that determines the infection rate.
<br>The reason why Daegu and Gyeongsangbuk-do have a high number of patients relative to their population are because infected **Sincheonji believers** had a huge prayer meeting, causing of exponential growth, and many of them traveled from Daegu to Gyeongsangbuk-do.

---
layout: post
title: Traffic Accidents in Korea 2018
subtitle : Analyzing traffic accident data from data.go.kr
tags: [Google_colab, Python, Pandas, Seaborn, Data Analysis]
author: Huey Kim
comments : False
---

# **traffic-accidents-in-korea-2018**

We will use [traffic accident dataset](https://www.data.go.kr/dataset/3038489/fileData.do) provided by Data.go.kr.

File Name: **도로교통공단_사고유형별_교통사고_통계(2018).zip**

## **List of Files**

**Dataset**: accident_type<br>
*Encoding*: EUC-KR

*   **byMonth.csv** (도로교통공단_사고유형별_월별_교통사고(2018).csv)
*   **byRoadType.csv** (도로교통공단_사고유형별_도로종류별_교통사고(2018).csv)
*   **bySuspectAge.csv** (도로교통공단_사고유형별_가해운전자_연령층별_교통사고(2018).csv)
*   **bySuspectCarType.csv** (도로교통공단_사고유형별_가해운전자_차종별_교통사고(2018).csv)
*   **bySuspectLawViolation.csv** (도로교통공단_사고유형별_가해운전자_법규위반별_교통사고(2018).csv)

Use this code to convert file with encoding **EUC-KR** to **UTF-8**.
```
iconv -f euc-kr -t utf-8 old.csv > new.csv
```

## **Importing Libraries**


```
%matplotlib inline

import os
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
```


```
# For Korean support
# Restart runtime if it doesn't work
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

%config InlineBackend.figure_format = 'retina'
!apt -qq -y install fonts-nanum

fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
fm._rebuild()

plt.rcParams['font.family'] = ['NanumBarunGothic']
sns.set_style("darkgrid", {"font.sans-serif":['NanumBarunGothic', 'Arial']})
```

    fonts-nanum is already the newest version (20170925-1).
    0 upgraded, 0 newly installed, 0 to remove and 25 not upgraded.


## **Setting Directory Path**


```
DIR_PATH = '/content/drive/My Drive/Colab Notebooks/data/kr-traffic-accidents-2018/accident_type/'
```

## **Data Analysis**

### **By Month**


```
byMonth = pd.read_csv(os.path.join(DIR_PATH, 'byMonth.csv'))
print('There are total {0} cases of monthly data \n'.format(byMonth.shape[0]))
byMonth.head()
```

    There are total 203 cases of monthly data 
    





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
      <th>사고유형대분류</th>
      <th>사고유형</th>
      <th>월</th>
      <th>발생건수</th>
      <th>사망자수</th>
      <th>부상자수</th>
      <th>중상</th>
      <th>경상</th>
      <th>부상신고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>01월</td>
      <td>1667</td>
      <td>88</td>
      <td>1707</td>
      <td>917</td>
      <td>720</td>
      <td>70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>02월</td>
      <td>1511</td>
      <td>62</td>
      <td>1533</td>
      <td>760</td>
      <td>713</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>03월</td>
      <td>1599</td>
      <td>76</td>
      <td>1629</td>
      <td>798</td>
      <td>751</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>04월</td>
      <td>1544</td>
      <td>62</td>
      <td>1578</td>
      <td>766</td>
      <td>750</td>
      <td>62</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>05월</td>
      <td>1495</td>
      <td>46</td>
      <td>1573</td>
      <td>739</td>
      <td>774</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```
# Types of accident
print(byMonth['사고유형'].describe())
```

    count     203
    unique     15
    top        기타
    freq       36
    Name: 사고유형, dtype: object


We can see that there are 15 unique types of accident in this dataset.


```
plt.figure(figsize=(10, 5))
sns.barplot(data=byMonth, x='월', y='발생건수', ci=None, estimator=sum)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f2ae6229e48>




![graph1](/assets/img/posts/p7_graph_1.png)



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


```
# Shows total injured people and number of people by each type
fig, ax = plt.subplots(figsize=(13, 5))
sns.set_color_codes("pastel")
sns.barplot(data=byMonth, x='부상자수', y='월', label='총 부상자 수', color='b', ci=None, estimator=sum)
sns.barplot(data=byMonth, x='경상', y='월', label='경상', color='g', ci=None, estimator=sum)
sns.barplot(data=byMonth, x='중상', y='월', label='중상', color='r', ci=None, estimator=sum)
sns.barplot(data=byMonth, x='사망자수', y='월', label='사망자 수', color='k', ci=None, estimator=sum)
ax.legend(ncol=2, loc='upper right', frameon=True, bbox_to_anchor=(1.25, 1))
ax.set(xlabel='부상자 수')
show_values_on_bars(ax, 'h', 200, 0.25)
```


![graph2](/assets/img/posts/p7_graph_2.png)


There were the most traffic accidents and most people were hurt in October 2018.

### **by RoadType**


```
byRoadType = pd.read_csv(os.path.join(DIR_PATH, 'byRoadType.csv'))
print('There are total {0} cases of data \n'.format(byRoadType.shape[0]))
byRoadType.head()
```

    There are total 118 cases of data 
    





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
      <th>사고유형대분류</th>
      <th>사고유형</th>
      <th>도로종류</th>
      <th>발생건수</th>
      <th>사망자수</th>
      <th>부상자수</th>
      <th>중상</th>
      <th>경상</th>
      <th>부상신고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>일반국도</td>
      <td>902</td>
      <td>108</td>
      <td>852</td>
      <td>490</td>
      <td>325</td>
      <td>37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>지방도</td>
      <td>1051</td>
      <td>93</td>
      <td>1017</td>
      <td>514</td>
      <td>463</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>특별광역시도</td>
      <td>8374</td>
      <td>271</td>
      <td>8679</td>
      <td>4253</td>
      <td>4019</td>
      <td>407</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>시도</td>
      <td>6829</td>
      <td>279</td>
      <td>6983</td>
      <td>3523</td>
      <td>3245</td>
      <td>215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>군도</td>
      <td>531</td>
      <td>24</td>
      <td>539</td>
      <td>300</td>
      <td>221</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```
print(byRoadType['도로종류'].describe())
```

    count     118
    unique      7
    top        시도
    freq       18
    Name: 도로종류, dtype: object



```
# x: Accident type, y: Accident count
plt.figure(figsize=(10, 5))
sns.barplot(data=byRoadType, x='사고유형', y='발생건수', ci=None, estimator=sum)
plt.xticks(rotation=45, ha='right')
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]),
     <a list of 15 Text major ticklabel objects>)




![graph3](/assets/img/posts/p7_graph_3.png)


We can see that **side collisions** take up the majority of accidents.


```
# Severity for each type of road
fig, ax2 = plt.subplots(figsize=(20, 5))
brt_side = byRoadType.loc[byRoadType['사고유형'] == '측면충돌']
brt_side2 = brt_side.drop(brt_side.columns[[0, 1, 8]], axis=1)
tidy = brt_side2.melt(id_vars='도로종류').rename(columns=str.title)
sns.barplot(data=tidy, x='도로종류', y='Value', hue='Variable', ax=ax2)
show_values_on_bars(ax2, 'v', modv=500)
```


![graph4](/assets/img/posts/p7_graph_4.png)


### **by Suspect Age**


```
bySuspectAge = pd.read_csv(os.path.join(DIR_PATH, 'bySuspectAge.csv'))
print('There are total {0} cases of data \n'.format(bySuspectAge.shape[0]))
bySuspectAge.head()
```

    There are total 147 cases of data 
    





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
      <th>사고유형대분류</th>
      <th>사고유형</th>
      <th>연령</th>
      <th>발생건수</th>
      <th>사망자수</th>
      <th>부상자수</th>
      <th>중상</th>
      <th>경상</th>
      <th>부상신고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>12세이하</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>13-20세</td>
      <td>521</td>
      <td>11</td>
      <td>616</td>
      <td>270</td>
      <td>304</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>21-30세</td>
      <td>2132</td>
      <td>87</td>
      <td>2247</td>
      <td>1116</td>
      <td>1032</td>
      <td>99</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>31-40세</td>
      <td>2751</td>
      <td>125</td>
      <td>2810</td>
      <td>1458</td>
      <td>1251</td>
      <td>101</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>41-50세</td>
      <td>3763</td>
      <td>176</td>
      <td>3804</td>
      <td>1905</td>
      <td>1789</td>
      <td>110</td>
    </tr>
  </tbody>
</table>
</div>




```
fig1, (ax3, ax4) = plt.subplots(ncols=2, figsize=(20, 5))
sns.barplot(data=bySuspectAge, x='연령', y='발생건수', ci=None, estimator=sum, ax=ax3)
# Accident type of age group 51~60
age_fs = bySuspectAge.loc[bySuspectAge['연령'] == '51-60세']
tidy = age_fs[['사고유형','부상자수']].melt(id_vars='사고유형').rename(columns=str.title)
sns.barplot(x='사고유형', y='Value', hue='Variable', data=tidy, ax=ax4, ci=None)
ax4.set(xlabel='사고유형 (51~60세)', ylabel='부상자수')

show_values_on_bars(ax3, 'v', modv=500)
show_values_on_bars(ax4, 'v', modv=500)
plt.xticks(rotation=45, ha='right')
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]),
     <a list of 15 Text major ticklabel objects>)




![graph5](/assets/img/posts/p7_graph_5.png)


Most suspects of car accident are from age group 51~60.

Number of side collisions were the greatest among accident types they caused.

### **by Suspect Car Type**


```
bySuspectCarType = pd.read_csv(os.path.join(DIR_PATH, 'bySuspectCarType.csv'))
print('There are total {0} cases of data \n'.format(bySuspectCarType.shape[0]))
bySuspectCarType.head()
```

    There are total 203 cases of data 
    





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
      <th>사고유형대분류</th>
      <th>사고유형</th>
      <th>차종</th>
      <th>발생건수</th>
      <th>사망자수</th>
      <th>부상자수</th>
      <th>중상</th>
      <th>경상</th>
      <th>부상신고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>승용차</td>
      <td>12697</td>
      <td>499</td>
      <td>12869</td>
      <td>6560</td>
      <td>5917</td>
      <td>392</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>승합차</td>
      <td>1185</td>
      <td>95</td>
      <td>1180</td>
      <td>661</td>
      <td>480</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>화물차</td>
      <td>2234</td>
      <td>156</td>
      <td>2183</td>
      <td>1222</td>
      <td>896</td>
      <td>65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>특수차</td>
      <td>40</td>
      <td>7</td>
      <td>35</td>
      <td>16</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>이륜차</td>
      <td>1288</td>
      <td>17</td>
      <td>1530</td>
      <td>667</td>
      <td>759</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>




```
print(bySuspectCarType['차종'].describe())
plt.figure(figsize=(10, 5))
sns.barplot(data=bySuspectCarType, x='차종', y='발생건수', ci=None, estimator=sum)
plt.xticks(rotation=45, ha='right')
```

    count     203
    unique     13
    top       승용차
    freq       18
    Name: 차종, dtype: object





    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
     <a list of 13 Text major ticklabel objects>)




![graph6](/assets/img/posts/p7_graph_6.png)


### **by Suspect Law Violation**


```
bySuspectLawViolation = pd.read_csv(os.path.join(DIR_PATH, 'bySuspectLawViolation.csv'))
print('There are total {0} cases of data \n'.format(bySuspectLawViolation.shape[0]))
bySuspectLawViolation.head()
```

    There are total 114 cases of data 
    





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
      <th>사고유형대분류</th>
      <th>사고유형</th>
      <th>법규위반</th>
      <th>발생건수</th>
      <th>사망자수</th>
      <th>부상자수</th>
      <th>중상</th>
      <th>경상</th>
      <th>부상신고</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>과속</td>
      <td>178</td>
      <td>100</td>
      <td>97</td>
      <td>78</td>
      <td>18</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>중앙선 침범</td>
      <td>187</td>
      <td>8</td>
      <td>183</td>
      <td>99</td>
      <td>80</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>신호위반</td>
      <td>2646</td>
      <td>70</td>
      <td>2836</td>
      <td>1384</td>
      <td>1358</td>
      <td>94</td>
    </tr>
    <tr>
      <th>3</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>안전거리 미확보</td>
      <td>13</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>차대사람</td>
      <td>횡단중</td>
      <td>안전운전 의무 불이행</td>
      <td>9021</td>
      <td>510</td>
      <td>9025</td>
      <td>4724</td>
      <td>3837</td>
      <td>464</td>
    </tr>
  </tbody>
</table>
</div>




```
plt.figure(figsize=(10, 5))
sns.barplot(data=bySuspectLawViolation, x='부상자수', y='법규위반',  label='부상자수', color='b', ci=None, estimator=sum)
sns.barplot(data=bySuspectLawViolation, x='경상', y='법규위반', label='경상', color='r', ci=None, estimator=sum)
sns.barplot(data=bySuspectLawViolation, x='중상', y='법규위반', label='중상', color='g', ci=None, estimator=sum)
plt.legend(ncol=2, loc='upper right', frameon=True)
```




    <matplotlib.legend.Legend at 0x7f2ae2ad7320>




![graph7](/assets/img/posts/p7_graph_7.png)


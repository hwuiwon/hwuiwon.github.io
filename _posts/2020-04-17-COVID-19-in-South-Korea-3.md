---
layout: post
title: COVID-19 in South Korea (3)
subtitle : Analyzing Patients and Floating Population
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

*   **PatientInfo.csv** (Epidemiological data of COVID-19 patients) ✔️
*   **PatientRoute.csv** (Route data of COVID-19 patients) ✔️
*   **SeoulFloating.csv** (Data of floating population in Seoul, South Korea (from SK Telecom Big Data Hub)) ✔️

## **By Patient**

### **Visualization**

#### **PatientInfo.csv**

Epidemiological data of COVID-19 patients

**Column Description**

Name | Description
:--- | :---
***patient_id*** | the ID of the patient
***global_num*** | the number given by KCDC
***sex*** | the sex of the patient
***birth_year*** | the birth year of the patient
***age*** | the age of the patient
***country*** | the country of the patient
***province*** | the province of the patient
***city*** | the city of the patient
***disease*** | TRUE: underlying disease / FALSE: no disease
***infection_case*** | the case of infection
***infection_order*** | the order of infection
***infected_by*** | the ID of who infected the patient
***contact_number*** | the number of contacts with people
***symptom_onset_date*** | the date of symptom onset
***confirmed_date*** | the date of being confirmed
***released_date*** | the date of being released
***deceased_date*** | the date of being deceased
***state*** | isolated / released / deceased


```
# Importing data: Patient info
patientinfo = pd.read_csv(os.path.join(DIR_PATH, 'PatientInfo.csv'))
p_total = len(patientinfo)
print('People mainly got infected by {0} ways'.format(len(patientinfo['infection_case'].unique())) + 
      ' and had {0} contacts per person in average.'.format(round(patientinfo['contact_number'].mean(), 2)))
print('There are {0} patient data in this set.'.format(p_total))

# Convert to Int64 to remove decimals and leave NaN
patientinfo['infected_by'] = patientinfo['infected_by'].astype('Int64')

# Show transpose of a matrix for better visualization
patientinfo.head().T
```

    People mainly got infected by 24 ways and had 18.91 contacts per person in average.
    There are 3128 patient data in this set.





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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>patient_id</th>
      <td>1000000001</td>
      <td>1000000002</td>
      <td>1000000003</td>
      <td>1000000004</td>
      <td>1000000005</td>
    </tr>
    <tr>
      <th>global_num</th>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>male</td>
      <td>male</td>
      <td>male</td>
      <td>male</td>
      <td>female</td>
    </tr>
    <tr>
      <th>birth_year</th>
      <td>1964</td>
      <td>1987</td>
      <td>1964</td>
      <td>1991</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>age</th>
      <td>50s</td>
      <td>30s</td>
      <td>50s</td>
      <td>20s</td>
      <td>20s</td>
    </tr>
    <tr>
      <th>country</th>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
    </tr>
    <tr>
      <th>province</th>
      <td>Seoul</td>
      <td>Seoul</td>
      <td>Seoul</td>
      <td>Seoul</td>
      <td>Seoul</td>
    </tr>
    <tr>
      <th>city</th>
      <td>Gangseo-gu</td>
      <td>Jungnang-gu</td>
      <td>Jongno-gu</td>
      <td>Mapo-gu</td>
      <td>Seongbuk-gu</td>
    </tr>
    <tr>
      <th>disease</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>infection_case</th>
      <td>overseas inflow</td>
      <td>overseas inflow</td>
      <td>contact with patient</td>
      <td>overseas inflow</td>
      <td>contact with patient</td>
    </tr>
    <tr>
      <th>infection_order</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>infected_by</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>2002000001</td>
      <td>&lt;NA&gt;</td>
      <td>1000000002</td>
    </tr>
    <tr>
      <th>contact_number</th>
      <td>75</td>
      <td>31</td>
      <td>17</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>symptom_onset_date</th>
      <td>2020-01-22</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-01-26</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>confirmed_date</th>
      <td>2020-01-23</td>
      <td>2020-01-30</td>
      <td>2020-01-30</td>
      <td>2020-01-30</td>
      <td>2020-01-31</td>
    </tr>
    <tr>
      <th>released_date</th>
      <td>2020-02-05</td>
      <td>2020-03-02</td>
      <td>2020-02-19</td>
      <td>2020-02-15</td>
      <td>2020-02-24</td>
    </tr>
    <tr>
      <th>deceased_date</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>state</th>
      <td>released</td>
      <td>released</td>
      <td>released</td>
      <td>released</td>
      <td>released</td>
    </tr>
  </tbody>
</table>
</div>




```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 2]})

patientinfo_gender = patientinfo[['patient_id', 'sex']].groupby('sex', as_index=False).count()
patientinfo_gender.columns = ['gender', 'count']

# Donut plot of confirmed cases by gender
ax1.title.set_text('Gender Distribution')
ax1.pie(patientinfo_gender['count'].values, labels=['female', 'male'],
        autopct='%.1f%%', startangle=90, pctdistance=0.85, colors=['#ff9999','#66b3ff'])
ax1.add_artist(plt.Circle((0, 0), 0.7, fc='white'))

# Age distribution of patients dataset
# 2 5 4 6 3 1
ax2.title.set_text('Age Distribution')
sns.countplot(data=patientinfo, x='age',
              order=patientinfo['age'].value_counts().index, ax=ax2)

plt.show()
```


![graph1](/assets/img/posts/p10_graph_1.png)



```
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))

# Where/How confirmed patients got infected
ax1.title.set_text('Route of Infection')
sns.countplot(data=patientinfo, y='infection_case', 
              order=patientinfo['infection_case'].value_counts().index, ax=ax1)
show_values_on_bars(ax1, 'h', 10, 0.25)

# Infection order of a patient
ax2.title.set_text('Infection Order of a Patient (excluding NaN)')
sns.countplot(data=patientinfo, x='infection_order',
              order=patientinfo['infection_order'].value_counts().index, ax=ax2)
show_values_on_bars(ax2, 'v', modv=0.2)

plt.show()
```


![graph2](/assets/img/posts/p10_graph_2.png)



```
transmit_order = patientinfo['infected_by'].value_counts().iloc[:10].index

# Top 10 patients who transmitted COVID-19 to others
fig, ax1 = plt.subplots(figsize=(10, 5))
plt.title('Top 10 patients who transmitted COVID-19')
sns.countplot(data=patientinfo, x='infected_by', order=transmit_order, ax=ax1)
plt.xticks(rotation=30, ha='right')
fig.show()
```


![graph3](/assets/img/posts/p10_graph_3.png)



```
# Information of top 10 COVID-19 carriers
transmit_order_df = patientinfo.loc[patientinfo['patient_id'].isin(transmit_order)]
transmit_order_df.T
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
      <th>124</th>
      <th>693</th>
      <th>1112</th>
      <th>1154</th>
      <th>1192</th>
      <th>1296</th>
      <th>1463</th>
      <th>1495</th>
      <th>1665</th>
      <th>1667</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>patient_id</th>
      <td>1000000125</td>
      <td>1200000031</td>
      <td>2000000125</td>
      <td>2000000167</td>
      <td>2000000205</td>
      <td>2000000309</td>
      <td>2000000476</td>
      <td>2000000508</td>
      <td>4100000006</td>
      <td>4100000008</td>
    </tr>
    <tr>
      <th>global_num</th>
      <td>7265</td>
      <td>31</td>
      <td>6780</td>
      <td>7663</td>
      <td>8100</td>
      <td>8632</td>
      <td>9742</td>
      <td>9928</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>female</td>
      <td>female</td>
      <td>male</td>
      <td>female</td>
      <td>female</td>
      <td>female</td>
      <td>female</td>
      <td>female</td>
      <td>female</td>
      <td>female</td>
    </tr>
    <tr>
      <th>birth_year</th>
      <td>1964</td>
      <td>1959</td>
      <td>1938</td>
      <td>1976</td>
      <td>1946</td>
      <td>1935</td>
      <td>1938</td>
      <td>1973</td>
      <td>1978</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>age</th>
      <td>50s</td>
      <td>60s</td>
      <td>80s</td>
      <td>40s</td>
      <td>70s</td>
      <td>80s</td>
      <td>80s</td>
      <td>40s</td>
      <td>40s</td>
      <td>40s</td>
    </tr>
    <tr>
      <th>country</th>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
      <td>Korea</td>
    </tr>
    <tr>
      <th>province</th>
      <td>Seoul</td>
      <td>Daegu</td>
      <td>Gyeonggi-do</td>
      <td>Gyeonggi-do</td>
      <td>Gyeonggi-do</td>
      <td>Gyeonggi-do</td>
      <td>Gyeonggi-do</td>
      <td>Gyeonggi-do</td>
      <td>Chungcheongnam-do</td>
      <td>Chungcheongnam-do</td>
    </tr>
    <tr>
      <th>city</th>
      <td>Nowon-gu</td>
      <td>NaN</td>
      <td>Seongnam-si</td>
      <td>Bucheon-si</td>
      <td>Seongnam-si</td>
      <td>Gunpo-si</td>
      <td>Uijeongbu-si</td>
      <td>Pyeongtaek-si</td>
      <td>Asan-si</td>
      <td>Cheonan-si</td>
    </tr>
    <tr>
      <th>disease</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>infection_case</th>
      <td>Guro-gu Call Center</td>
      <td>Shincheonji Church</td>
      <td>etc</td>
      <td>contact with patient</td>
      <td>contact with patient</td>
      <td>etc</td>
      <td>etc</td>
      <td>overseas inflow</td>
      <td>contact with patient</td>
      <td>gym facility in Cheonan</td>
    </tr>
    <tr>
      <th>infection_order</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>infected_by</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1000000125</td>
      <td>1000000138</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>4100000007</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>contact_number</th>
      <td>NaN</td>
      <td>1160</td>
      <td>3</td>
      <td>NaN</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41</td>
      <td>130</td>
    </tr>
    <tr>
      <th>symptom_onset_date</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-03-18</td>
      <td>2020-03-29</td>
      <td>NaN</td>
      <td>2020-02-22</td>
      <td>2020-02-20</td>
    </tr>
    <tr>
      <th>confirmed_date</th>
      <td>2020-03-08</td>
      <td>2020-02-18</td>
      <td>2020-03-07</td>
      <td>2020-03-10</td>
      <td>2020-03-14</td>
      <td>2020-03-19</td>
      <td>2020-03-30</td>
      <td>2020-04-01</td>
      <td>2020-02-26</td>
      <td>2020-02-26</td>
    </tr>
    <tr>
      <th>released_date</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020-03-22</td>
      <td>2020-03-29</td>
    </tr>
    <tr>
      <th>deceased_date</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>state</th>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>isolated</td>
      <td>released</td>
      <td>released</td>
    </tr>
  </tbody>
</table>
</div>




```
# Days took to release prior positive patients (Exclude NaN values)
patientinfo_release = pd.DataFrame()
patientinfo_release['c_date'] = pd.to_datetime(patientinfo['confirmed_date'], format='%Y-%m-%d')
patientinfo_release['r_date'] = pd.to_datetime(patientinfo['released_date'], format='%Y-%m-%d')
patientinfo_release['days_took'] = (patientinfo_release['r_date']
                                    - patientinfo_release['c_date']).dt.days.astype('Int64')
patientinfo_release = patientinfo_release.dropna()

plt.figure(figsize=(10, 3))
plt.title('Days took to get released')
sns.boxplot(x=patientinfo_release['days_took'])
# sns.swarmplot(x=patientinfo_release['days_took'], color='.25')
plt.show()
```


![graph4](/assets/img/posts/p10_graph_4.png)



```
p_nosymp = patientinfo['symptom_onset_date'].isna().sum()

# Proportion of patients with/without symptom
plt.figure(figsize=(7, 7))
plt.title('Patients with Symptom')
plt.pie([p_total - p_nosymp, p_nosymp], labels=[f'Yes ({p_total - p_nosymp})', f'No ({p_nosymp})'], 
        autopct='%.1f%%', pctdistance=0.85)
plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, fc='white'))
plt.show()
```


![graph5](/assets/img/posts/p10_graph_5.png)


#### **PatientRoute.csv**

Route data of COVID-19 patients


```
# Importing data: Patient route
patientroute = pd.read_csv(os.path.join(DIR_PATH, 'PatientRoute.csv'))
patientroute.head()
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
      <th>patient_id</th>
      <th>global_num</th>
      <th>date</th>
      <th>province</th>
      <th>city</th>
      <th>type</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000000001</td>
      <td>2.0</td>
      <td>2020-01-22</td>
      <td>Gyeonggi-do</td>
      <td>Gimpo-si</td>
      <td>airport</td>
      <td>37.615246</td>
      <td>126.715632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1000000001</td>
      <td>2.0</td>
      <td>2020-01-24</td>
      <td>Seoul</td>
      <td>Jung-gu</td>
      <td>hospital</td>
      <td>37.567241</td>
      <td>127.005659</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000000002</td>
      <td>5.0</td>
      <td>2020-01-25</td>
      <td>Seoul</td>
      <td>Seongbuk-gu</td>
      <td>etc</td>
      <td>37.592560</td>
      <td>127.017048</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1000000002</td>
      <td>5.0</td>
      <td>2020-01-26</td>
      <td>Seoul</td>
      <td>Seongbuk-gu</td>
      <td>store</td>
      <td>37.591810</td>
      <td>127.016822</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1000000002</td>
      <td>5.0</td>
      <td>2020-01-26</td>
      <td>Seoul</td>
      <td>Seongdong-gu</td>
      <td>public_transportation</td>
      <td>37.563992</td>
      <td>127.029534</td>
    </tr>
  </tbody>
</table>
</div>




```
patientroute_top_log = pd.DataFrame(patientroute['patient_id'].value_counts().head(10))
print('There are {0} patients\' route data.'.format(len(patientroute['patient_id'].unique())))
patientroute_top_place = pd.DataFrame(patientroute['type'].value_counts().head(10))
patientroute_top_place.reset_index(level=0, inplace=True)
patientroute_top_place.columns = ['type', 'count']
patientroute_top_place
```

    There are 939 patients' route data.





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
      <th>type</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>etc</td>
      <td>1698</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hospital</td>
      <td>1496</td>
    </tr>
    <tr>
      <th>2</th>
      <td>store</td>
      <td>507</td>
    </tr>
    <tr>
      <th>3</th>
      <td>restaurant</td>
      <td>451</td>
    </tr>
    <tr>
      <th>4</th>
      <td>public_transportation</td>
      <td>382</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pharmacy</td>
      <td>200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>airport</td>
      <td>120</td>
    </tr>
    <tr>
      <th>7</th>
      <td>church</td>
      <td>120</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cafe</td>
      <td>85</td>
    </tr>
    <tr>
      <th>9</th>
      <td>school</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.title.set_text('Top 10 Places COVID-19 Patients Visited')
sns.barplot(data=patientroute_top_place, x='type', y='count', ax=ax1)
show_values_on_bars(ax1, 'v', modv=20)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.show()
```


![graph6](/assets/img/posts/p10_graph_6.png)



```
# Route of top 10 patients who spread COVID-19
patient_routes = []

for i in range(len(transmit_order)):
    a = []
    tmp_route = patientroute.loc[patientroute['patient_id'] == 
                                 transmit_order[i]].reset_index(drop=True)
    for j in range(len(tmp_route)):
        a.append(tuple([tmp_route.loc[j].latitude, tmp_route.loc[j].longitude]))
    patient_routes.append(a)

print('Saved in \'patient_routes\'')
```

    Saved in 'patient_routes'



```
route_southKR = folium.Map(location=[36.5, 128], tiles="cartodbpositron",
                         zoom_start=8, min_zoom=5)
folium.Choropleth(geo_data=os.path.join(DIR_PATH, 'province_geo.json'),
                  fill_color='#ffff66', line_opacity=0.5, fill_opacity=0.3).add_to(route_southKR)

for i in range(len(patient_routes)):
    for places in patient_routes[i]:
        folium.Marker(places).add_to(route_southKR)
    ran_c = list(np.random.choice(range(256), size=3))
    folium.PolyLine(patient_routes[i], color='#%02x%02x%02x' %
                    (ran_c[0], ran_c[1], ran_c[2])).add_to(route_southKR)

route_southKR
```


### **Analysis & Conclusion**

1. Majority of patients got infected by contacting with patient and by overseas inflow.
2. Top patients who infected others spread the disease by working/attending chapel in a crowded place and visited many places before they were tested positive.
3. It took approximately 20 days for patients to get released after testing positive. 
4. 85.7% of patients had no symptoms before they got tested.

Considering the mean value of days took to get released is 20, we can infer that most of the patients take 20 days to recover from COVID-19. As there are a high percentage of people who were tested positive without any symptoms, we can see that they took appropriate procedures to prevent infection from spreading. The patient route data provided by the government might have encouraged people to get tested.


## **By Floating Population (Seoul)**

### **Assumption**

The provinces with higher floating population will have higher confirmed COVID-19 cases.

### **Visualization**


```
seoul_float = pd.read_csv(os.path.join(DIR_PATH, 'SeoulFloating.csv'))
seoul_float
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
      <th>hour</th>
      <th>birth_year</th>
      <th>sex</th>
      <th>province</th>
      <th>city</th>
      <th>fp_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>0</td>
      <td>20</td>
      <td>female</td>
      <td>Seoul</td>
      <td>Dobong-gu</td>
      <td>19140</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-01</td>
      <td>0</td>
      <td>20</td>
      <td>male</td>
      <td>Seoul</td>
      <td>Dobong-gu</td>
      <td>19950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-01-01</td>
      <td>0</td>
      <td>20</td>
      <td>female</td>
      <td>Seoul</td>
      <td>Dongdaemun-gu</td>
      <td>25450</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-01-01</td>
      <td>0</td>
      <td>20</td>
      <td>male</td>
      <td>Seoul</td>
      <td>Dongdaemun-gu</td>
      <td>27050</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-01-01</td>
      <td>0</td>
      <td>20</td>
      <td>female</td>
      <td>Seoul</td>
      <td>Dongjag-gu</td>
      <td>28880</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>431995</th>
      <td>2020-02-29</td>
      <td>23</td>
      <td>70</td>
      <td>male</td>
      <td>Seoul</td>
      <td>Yangcheon-gu</td>
      <td>12080</td>
    </tr>
    <tr>
      <th>431996</th>
      <td>2020-02-29</td>
      <td>23</td>
      <td>70</td>
      <td>female</td>
      <td>Seoul</td>
      <td>Yeongdeungpo-gu</td>
      <td>17750</td>
    </tr>
    <tr>
      <th>431997</th>
      <td>2020-02-29</td>
      <td>23</td>
      <td>70</td>
      <td>male</td>
      <td>Seoul</td>
      <td>Yeongdeungpo-gu</td>
      <td>13290</td>
    </tr>
    <tr>
      <th>431998</th>
      <td>2020-02-29</td>
      <td>23</td>
      <td>70</td>
      <td>female</td>
      <td>Seoul</td>
      <td>Yongsan-gu</td>
      <td>12590</td>
    </tr>
    <tr>
      <th>431999</th>
      <td>2020-02-29</td>
      <td>23</td>
      <td>70</td>
      <td>male</td>
      <td>Seoul</td>
      <td>Yongsan-gu</td>
      <td>8560</td>
    </tr>
  </tbody>
</table>
<p>432000 rows × 7 columns</p>
</div>




```
# Data of patients who live in Seoul
patientinfo_seoul = patientinfo[patientinfo['province'] == 'Seoul']
p_by_province = pd.DataFrame(patientinfo_seoul['city'].value_counts()).reset_index(level=0)
p_by_province.columns = ['district', 'count']
p_by_province.head()
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
      <th>district</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gangnam-gu</td>
      <td>53</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Gwanak-gu</td>
      <td>41</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Songpa-gu</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guro-gu</td>
      <td>34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dongdaemun-gu</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 16))

# Number of Patients in District (Seoul)
ax1.title.set_text('Number of Patients in Seoul (by District)')
sns.barplot(data=p_by_province, x='district', y='count', ax=ax1)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Floating Population in Seoul
ax2.title.set_text('Floating Population in Seoul (by District)')
sns.lineplot(data=seoul_float, x='date', y='fp_num', hue='city', ax=ax2)

ax2.xaxis.set_major_locator(ticker.MultipleLocator(base=6))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
plt.legend(loc='upper left', ncol=4)

fig.tight_layout()
plt.show()
```


![graph7](/assets/img/posts/p10_graph_7.png)


### **Analysis & Conclusion**

1. Provinces in Seoul show a similar trend of floating population.
2. There were no significant change in the floating population before and after COVID-19 situation between Jan 1, 2020 and Feb 29, 2020.
3. There was a 'unknown' spike in floating population in Feb 23, 2020.

It seems floating population is not a major factor that decides infection.

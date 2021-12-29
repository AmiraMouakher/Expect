import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

def load (path) : 
  df = pd.read_csv(path)
  return df

def nan_viz (data) :
  total = data.isnull().sum().sort_values(ascending=False)


  percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

  missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
  f, ax = plt.subplots(figsize=(15, 6))
  plt.xticks(rotation='90')
  sns.barplot(x=missing_data.index, y=missing_data['Percent'])
  plt.xlabel('Features', fontsize=15)
  plt.ylabel('Percent of Missing Values', fontsize=15)
  plt.title('Percentage of Missing Data by Feature', fontsize=15)
  missing_data.head()

def random_sampling(df, variable): 
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0,replace=True)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable] = random_sample
    return(df)

def prepare_compteur (cons,i) :
  df = pd.DataFrame(cons.iloc[i][1:])
  df.reset_index(level=0, inplace=True)
  df.columns = ['Datetime','consommation']
  # df = df.set_index('Datetime')
  return (df)

def feature_weekend(row): 

    if row["day"] == 5 or row["day"] == 6 or row["day"] == 7:
        return "Weekend"
    else :
      return "NonWeekend"
import copy
from tqdm import tqdm
def time_features (df) : 
    df_copy = df.copy()
    df= df.set_index('Datetime')
    df1 = pd.DataFrame([], columns=["year","month", "day", "hour","minute"])
    from datetime import datetime
    print('Preparing Time features : Please Wait')
    for i in tqdm(range(len(df.index))):
        date = datetime.strptime(df.index[i] , '%Y-%m-%d %H:%M:%S' )
        df_tmp = pd.DataFrame([(date.year,date.month ,date.day, date.hour , date.minute)], columns=["year","month", "day", "hour","minute"])
        df1 = df1.append(df_tmp)
    df1 = df1.reset_index()
    df1.drop('index',axis=1,inplace=True)
    df_train = pd.concat([df_copy,df1],axis= 1,ignore_index=True )
    df_copy = df_train
    df_copy.columns = ["Datetime" ,"consommation" ,"year","month", "day", "hour","minute"]
    df_copy['season'] = df_copy['month'].apply(lambda month_number: (month_number%12 + 3)//3)
    df_copy['dayofyear'] = [pd.to_datetime(df.index[i]).timetuple().tm_yday for i in range(len(df_copy)) ]
    df_copy['day_string'] = [pd.to_datetime(df_copy['Datetime'][i]).strftime("%A") for i in range(len(df_copy)) ]
    df_copy['week_of_year'] =  pd.to_datetime(df_copy['Datetime']).dt.weekofyear
    df_copy["weekend"] = [ feature_weekend(df_copy.iloc[i]) for i in range(len(df_copy)) ]
    return(df_copy)

def casting(df) : # casting
      df['hour'] = df['hour'].astype('int64')
      df['month'] = df['month'].astype('int64')
      df['consommation'] = df['consommation'].astype('float')

def mapping (df) : #mapping
    df['weekend'] = df['weekend'].map({'NonWeekend':0, 'Weekend':1})

def encoding (df) : # encodage
  from sklearn import preprocessing
  le = preprocessing.LabelEncoder()
  df['day_string'] = le.fit_transform(df['day_string'])

# plotting

def plot_time_serie (df) :
  fig = px.line(df,
                x='Datetime',
                y='consommation',
                title=f'consommation en fonction du temps')
  fig.show()


def aggregated_plot(df, agg1, agg2, attribute, title):
    _ = df \
        .groupby([agg1, agg2], as_index=False) \
        .agg({'consommation': attribute})

    fig = px.line(_,
                  x=agg1,
                  y='consommation',
                  color=agg2,
                  title=title)
    fig.show()

def weather_avg_feat(weather_avg,df) :
    temp = weather_avg[weather_avg['meter_id']== cons.iloc[3001]['meter_id']].transpose()[1:]
    temp.columns=['weather']
    L =[]
    for x in temp['weather'] :
      for i in range(48) :
        L.append(x)
    df['weather_avg'] = L

def weather_min_feat(weather_avg,df) :
    temp = weather_min[weather_min['meter_id']== cons.iloc[3001]['meter_id']].transpose()[1:]
    temp.columns=['weather']
    L =[]
    for x in temp['weather'] :
      for i in range(48) :
        L.append(x)
    df['weather_min'] = L

def weather_max_feat(weather_avg,df) :
    temp = weather_max[weather_avg['meter_id']== cons.iloc[3001]['meter_id']].transpose()[1:]
    temp.columns=['weather']
    L =[]
    for x in temp['weather'] :
      for i in range(48) :
        L.append(x)
    df['weather_max'] = L

def Elbow_curve(df) :
  scaler = MinMaxScaler()
  weather_scaled = scaler.fit_transform(df[['weather_min','weather_avg','weather_max','season']])
  # optimum K
  from sklearn.cluster import KMeans

  Nc = range(1, 20)
  kmeans = [KMeans(n_clusters=i) for i in Nc]
  kmeans

  score = [kmeans[i].fit(weather_scaled).score(weather_scaled) for i in range(len(kmeans))]
  score
  plt.plot(Nc,score)
  plt.xlabel('Number of Clusters')
  plt.ylabel('Score')
  plt.title('Elbow Curve')
  plt.show()
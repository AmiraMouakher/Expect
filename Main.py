from Utils import  *
import warnings
warnings.filterwarnings('ignore')


def main():
    print('loading required files')
    cons = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\consumption.csv")
    weather_max = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-max.csv")
    weather_min = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-min.csv")
    weather_avg = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-avg.csv")
    add = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\addInfo.csv")
    print('Nans pourcentage visualizations ')
    nan_viz(cons.loc[:, '2017-01-01 00:00:00':'2017-01-01 23:30:00']) 
    nan_viz(weather_min.loc[:, '2017-01-01 00:00:00':'2017-01-05 00:00:00'])
    nan_viz(weather_avg.loc[:, '2017-01-01 00:00:00':'2017-01-05 00:00:00'])
    nan_viz(weather_max.loc[:, '2017-01-01 00:00:00':'2017-01-05 00:00:00'])
    nan_viz(add)
    print('visualization plots')
    df = random_sampling(prepare_compteur(cons, 3001), 'consommation')
    df = time_features(df)
    casting(df)
    mapping(df)
    encoding(df)
    Features = ['minute', 'day', 'day_string', 'dayofyear', 'month', 'season']
    for x in Features: 
        aggregated_plot(df, 'hour', x, 'median', 'Median Hourly Power Demand per ' + x)
    print('Elbow curve for weather clustering')
    weather_min_feat(weather_avg, df)
    weather_avg_feat(weather_avg, df)
    weather_max_feat(weather_avg, df)
    Elbow_curve(df)


if __name__ == "__main__":
    main()
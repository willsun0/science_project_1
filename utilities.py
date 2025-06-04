import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the uploaded Excel file
def load_water_data(file_path):
    # file_path = "../Data/Complete_Data_WQI.xlsx"
    Complete_Data_WQI = pd.read_excel(file_path)

    print(Complete_Data_WQI.shape)
    print(Complete_Data_WQI['WQS'].value_counts(dropna=False))
    print(Complete_Data_WQI['WQS2'].value_counts(dropna=False))
    print('% of Bad water samples: ',Complete_Data_WQI[Complete_Data_WQI['WQS2']=='Bad'].shape[0]/Complete_Data_WQI.shape[0])


    print('===== After dropping WQS and WQS2 ========')
    df = Complete_Data_WQI.copy()

    df.drop(['WQS','WQS2'], axis=1, inplace=True)
    df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'], format='%Y-%m-%d')


    # Show basic info and first few rows
    # df.info(), df.head()
    print('Shape of dataset: ',df.shape)
    # print(df['WQS'].value_counts(dropna=False))
    # print(df['WQS2'].value_counts(dropna=False))
    # print('% of Bad water samples: ',df[df['WQS2']=='Bad'].shape[0]/df.shape[0])
    return df


def split_scale_data(df):
    # Define the features (excluding 'WQS' and 'WQS2')
    # features = ['Sulfate (mg/L)', 'Chloride (mg/L)', 'Sodium (mg/L)', 'Potassium (mg/L)',
    #                 'Calcium (mg/L)', 'Magnesium (mg/L)', 'Total Dissolved Solids (mg/L)',
    #                 'Turbidity (NTU)', 'Temperature (deg C)', 'pH',
    #                 'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Fecal Coliform (cfu/100ml)']
    features = [x for x in df.columns if ((x!='ActivityStartDate') & (x!='WQI'))]

    target = 'WQI'

    X = df[features]
    y = df[target]
    dates = df['ActivityStartDate']

    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # dates_test = df['ActivityStartDate'].iloc[split_idx:]
    dates_trian, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dates_trian, dates_test



def split_data(df):
    # Define the features (excluding 'WQS' and 'WQS2')
    # features = ['Sulfate (mg/L)', 'Chloride (mg/L)', 'Sodium (mg/L)', 'Potassium (mg/L)',
    #                 'Calcium (mg/L)', 'Magnesium (mg/L)', 'Total Dissolved Solids (mg/L)',
    #                 'Turbidity (NTU)', 'Temperature (deg C)', 'pH',
    #                 'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Fecal Coliform (cfu/100ml)']
    features = [x for x in df.columns if ((x!='ActivityStartDate') & (x!='WQI'))]
    print('features: ', features)

    # selected_features = ['Temperature (deg C)',
    #                 'Dissolved Oxygen (mg/L)', 
    #                 'Turbidity (NTU)',
    #             #  'Biochemical Oxygen Demand (mg/L)',
    #                 'Total Dissolved Solids (mg/L)',
    #                 'Fecal Coliform (cfu/100ml)', 
    #                 'pH', 
    #                 'Sulfate (mg/L)'
    #                 ]
    target = 'WQI'

    X = df[features]
    y = df[target]
    dates = df['ActivityStartDate']

    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    # dates_test = df['ActivityStartDate'].iloc[split_idx:]
    dates_trian, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    # X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape

    return X_train, X_test, y_train, y_test, dates_trian, dates_test



def scale_data(X_train, X_test):
    # Normalize features
    scaler = StandardScaler()
    scaler.fit(X_train)
    # X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_test_scaled


def create_lag_features(df, lag_num_days):
    features = [x for x in df.columns if ((x!='ActivityStartDate') & (x!='WQI'))]

    df_lag = df.copy()
    for lag in range(1, lag_num_days):
        for col in features:
            df_lag[f'{col}_lag{lag}'] = df_lag[col].shift(lag)
    df_lag.dropna(inplace=True)

    return df_lag
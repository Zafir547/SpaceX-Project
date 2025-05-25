import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    df = pd.read_csv("data/enriched_spacex_launches.csv")

    # Drop missing target labels
    df = df[df['success'].notna()]

    # Select relevant features
    df = df[[
        'name',
        'rocket',
        'date_utc',
        'success',
        'payloads',
        'launchpad',
        'temperature',
        'humidity',
        'wind_speed'
    ]]

    # Fill missing weather values
    df[['temperature', 'humidity', 'wind_speed']] = df[['temperature', 'humidity', 'wind_speed']].fillna(method='ffill')

    # Convert launch date to datetime features
    df['date_utc'] = pd.to_datetime(df['date_utc'])
    df['year'] = df['date_utc'].dt.year
    df['month'] = df['date_utc'].dt.month
    df['day'] = df['date_utc'].dt.day
    df['hour'] = df['date_utc'].dt.hour

    # Encode categorical variables
    le_rocket = LabelEncoder()
    le_launchpad = LabelEncoder()

    df['rocket_encoded'] = le_rocket.fit_transform(df['rocket'].astype(str))
    df['launchpad_encoded'] = le_launchpad.fit_transform(df['launchpad'].astype(str))

    # Final feature set
    df_final = df[[
        'rocket_encoded',
        'launchpad_encoded',
        'temperature',
        'humidity',
        'wind_speed',
        'year',
        'month',
        'day',
        'hour',
        'success'
    ]]

    df_final.to_csv("data/processed_spacex_data.csv", index=False)
    print("Processed data saved.")

if __name__ == "__main__":
    preprocess_data()

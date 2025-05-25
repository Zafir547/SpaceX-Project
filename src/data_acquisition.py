import requests
import pandas as pd
import time

def fetch_spacex_data():
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    launches = response.json()
    print(f"Fetched {len(launches)} launches.")

    df = pd.json_normalize(launches)
    df.to_csv("data/raw_spacex_launches.csv", index=False)
    print("Raw launch data saved.")
    return df

if __name__ == "__main__":
    fetch_spacex_data() 
import requests
import pandas as pd
import time

API_KEY = "YOUR-API-KEY"
CURRENT_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
HISTORICAL_WEATHER_URL = "http://api.openweathermap.org/data/2.5/onecall/timemachine"

class WeatherFetchError(Exception):
    """Custom exception for weather fetch failures."""
    pass

def fetch_current_weather(city: str) -> dict:
    """
    Fetch current weather for a given city name.
    """
    try:
        params = {
            "q": city,
            "units": "metric",
            "appid": API_KEY
        }
        resp = requests.get(CURRENT_WEATHER_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
        weather = data["weather"][0]
        main = data["main"]
        wind = data["wind"]

        return {
            "city": data["name"],
            "temperature": main["temp"],
            "humidity": main["humidity"],
            "wind_speed": wind["speed"],
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }

    except Exception as e:
        print(f"[Error] Failed to fetch current weather for '{city}': {e}")
        return {
            "city": city,
            "temperature": None,
            "humidity": None,
            "wind_speed": None,
            "lat": None,
            "lon": None
        }

def fetch_historical_weather(lat: float, lon: float, timestamp: int) -> dict:
    """
    Fetch historical weather using latitude, longitude, and UNIX timestamp.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "dt": timestamp,
        "appid": API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(HISTORICAL_WEATHER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        current = data["current"]
        return {
            "temperature": current["temp"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_speed"]
        }
    except Exception as e:
        print(f"[Error] Failed to fetch historical weather: {e}")
        return {
            "temperature": None,
            "humidity": None,
            "wind_speed": None
        }

def enrich_launch_data_with_weather():
    df = pd.read_csv("data/raw_spacex_launches.csv")
    df = df[df['success'].notna()]  # remove rows with missing success

    enriched_rows = []

    for _, row in df.iterrows():
        city = row.get("location") or "New York"  # default fallback city
        timestamp = int(row.get("date_unix", 0))

        # First, get coordinates from the city name
        current_weather = fetch_current_weather(city)
        lat = current_weather.get("lat")
        lon = current_weather.get("lon")

        # Then, use lat/lon for historical weather at launch time
        if lat is not None and lon is not None and timestamp > 0:
            hist_weather = fetch_historical_weather(lat, lon, timestamp)
        else:
            hist_weather = {"temperature": None, "humidity": None, "wind_speed": None}

        enriched_rows.append({
            **row,
            "city": city,
            "temperature": hist_weather["temperature"],
            "humidity": hist_weather["humidity"],
            "wind_speed": hist_weather["wind_speed"]
        })

        time.sleep(1)  # avoid rate-limiting

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv("data/enriched_spacex_launches.csv", index=False)
    print("✅ Enriched data with weather saved to 'data/enriched_spacex_launches.csv'.")

if __name__ == "__main__":
    enrich_launch_data_with_weather()

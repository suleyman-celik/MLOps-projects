
import os
import pickle
import zipfile
import requests

from glob import glob

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

def fetch_data(raw_data_path: str, location: str, year: int, month: int) -> None:
    """Fetches data from the NYC Citi Bike dataset and saves it locally"""
    filename = f'{location}{year}{month:0>2}-citibike-tripdata.csv.zip'
    filepath = os.path.join(raw_data_path, filename)
    url = f'https://s3.amazonaws.com/tripdata/{filename}'

    # Create the destination folder if it does not exist
    os.makedirs(raw_data_path, exist_ok=True)

    # Download the data
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as f_out:
            f_out.write(response.content)
    else:
        print(f"Failed to download data for {location} {year}-{month:02d}. HTTP Status Code: {response.status_code}")
        return

    # Extract the CSV file from the ZIP file
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        # Filter to get only the relevant CSV files
        csv_files = [file for file in zip_ref.namelist() if file.endswith('.csv') and not file.startswith('__MACOSX')]

        if len(csv_files) != 1:
            print(f"Unexpected file structure in {filename}. Found files: {csv_files}")
            return

        # Extract the CSV file to the specified directory
        zip_ref.extract(csv_files[0], path=raw_data_path)
        print(f"Extracted {csv_files[0]} to {raw_data_path}")

def download_data(raw_data_path: str, locations: list, years: list, months: list) -> None:
    try:
        # Download data for each combination of location, year, and month
        for loc in locations: 
            for year in years:       
                for month in months:
                    print(f"Fetching data for {loc} {year}-{month:02d}...")
                    fetch_data(raw_data_path, loc, year, month)
    except Exception as e:
        print("An error occurred during the data download process:", e)


def read_data(file_name: str)-> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_csv(file_name)

    # Convert Datetime
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at']   = pd.to_datetime(df['ended_at'])

    df["duration"] = df["ended_at"]-df["started_at"]
    df["duration_minutes"] = df["duration"].dt.total_seconds()/60

    # define criteria for outliers
    lower_threshold = 1
    upper_threshold = 60

    # filter dataframe based on threshold
    df = df[
    (df["duration_minutes"]>=lower_threshold) &
    (df["duration_minutes"]<=upper_threshold)
    ]

    # Define the categorical columns
    categorical_features = [
        'start_station_id',
        'end_station_id'
    ]
    
    df[categorical_features] = df[categorical_features].astype(str)
    # print(df.shape)
    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer = None, fit_dv: bool = False):
    def haversine_distance(row):
        lat1, lon1, lat2, lon2 = row['start_lat'], row['start_lng'], row['end_lat'], row['end_lng']
        # Convert latitude and longitude from degrees to radians
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])

        # Radius of the Earth in kilometers
        radius = 6371.0

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c    
        return distance

    """Add features to the model"""
    # Add location ID
    df['start_to_end_station_id'] = df['start_station_id'] + '_' + df['end_station_id']
    categorical = ["start_to_end_station_id"]

    # Calc Distance
    df['trip_distance'] = df.apply(haversine_distance, axis=1).fillna(0)
    numerical   = ['trip_distance']
    dicts       = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        # return sparse matrix
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
        
    # Convert X the sparse matrix  to pandas DataFrame, but too slow
    # X = pd.DataFrame(X.toarray(), columns=dv.get_feature_names_out())
    # X = pd.DataFrame.sparse.from_spmatrix(X, columns=dv.get_feature_names_out())

    try:
        # Extract the target
        target = 'member_casual'
        y = df[target].values
    except Exception as e:
        print("In preprocess Something Wrong...", e)
        pass
    # print(X.shape, y.shape)
    return (X, y), dv

def dump_pickle(obj, filename: str, dest_path: str): 
    file_path = os.path.join(dest_path, filename)
       
    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)
    with open(file_path, "wb") as f_out:
        return pickle.dump(obj, f_out)  


def run_data_prep(raw_data_path = "./data", dest_path= "./output",location = "JC-", years = "2024", months = "2 3 4")->None:
    # parameters
    locations = location.split(",")
    years = [int(year) for year in years.split()]
    months = [int(month) for month in months.split()]
    # print(locations, years, months)

    # download data
    print("data downloading...")
    download_data(raw_data_path, locations, years, months)
    print("data downloaded!")
    # print(sorted(glob(f'./data/*')))

    # Load csv files
    df_train = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[0]:0>2}-citibike-tripdata.csv')
    )
    df_val = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[1]:0>2}-citibike-tripdata.csv')
    )
    df_test = read_data(
        os.path.join(raw_data_path, f'{locations[0]}{years[0]}{months[2]:0>2}-citibike-tripdata.csv')
    )

    # Fit the DictVectorizer and preprocess data
    (X_train, y_train), dv = preprocess(df_train,fit_dv=True)
    (X_test, y_test), _ = preprocess(df_test, dv)
    (X_val, y_val), _ = preprocess(df_test, dv)

    # Save DictVectorizer and datasets
    dump_pickle(dv, "dv.pkl", dest_path)
    dump_pickle((X_train, y_train), "train.pkl",dest_path)
    dump_pickle((X_test, y_test), "test.pkl", dest_path)
    dump_pickle((X_val, y_val), "val.pkl", dest_path)

if __name__ == "__main__":
    run_data_prep()

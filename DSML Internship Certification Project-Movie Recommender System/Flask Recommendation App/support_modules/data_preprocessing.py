import os
import pandas as pd

def load_data():
    current_main_dir = os.path.abspath('..')
    movies_df = pd.read_csv(current_main_dir + "/data/movies.csv")
    ratings_df = pd.read_csv(current_main_dir + "/data/ratings.csv")
    return movies_df, ratings_df

def preprocess_data(movies_df):
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    # Remove year
    movies_df['title'] = movies_df['title'].transform(lambda item: item[0:-7])
    return movies_df

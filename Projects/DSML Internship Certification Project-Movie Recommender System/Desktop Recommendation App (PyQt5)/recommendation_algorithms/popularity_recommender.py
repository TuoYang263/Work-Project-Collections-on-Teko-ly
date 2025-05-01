import pandas as pd

def popularity_recommender(movies_df, ratings_df, genre, min_reviews, top_n):
    # Filter by genre (given in the parameter)
    genre_movies = movies_df[movies_df['genres_list'].apply(lambda g: genre in g)]
    # Merge with ratings (connect movies with ratings)
    merged = pd.merge(ratings_df, genre_movies, on='movieId')
    # Aggregate data
    grouped = merged.groupby('title').agg(
        avg_rating=('rating', lambda x: round(x.mean(), 6)),
        review_count=('rating', 'count')
    ).reset_index()
    # Filter by min_reviews
    filtered = grouped[grouped['review_count'] >= min_reviews]
    # Sort by avg_rating
    sorted_movies = filtered.sort_values(by='avg_rating', ascending=False).head(top_n)
    # Adding a sequential number column and make it to the 1st position
    sorted_movies = sorted_movies.set_index(pd.Series(range(1, len(sorted_movies) + 1))).reset_index()
    # Rename the column names of the dataframe sorted_movies
    sorted_movies = sorted_movies.rename(columns={'index': 'S.No',
                                                  'title': 'Movie Title',
                                                  'avg_rating': 'Average Movie Rating',
                                                  'review_count': 'Num Reviews'})
    return sorted_movies
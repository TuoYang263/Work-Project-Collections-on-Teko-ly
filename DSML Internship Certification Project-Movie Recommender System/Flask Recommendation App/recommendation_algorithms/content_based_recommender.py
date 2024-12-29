import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def content_based_recommender(movies_df, title, top_n):
    # Prepare genre data
    movies_df['genres_str'] = movies_df['genres_list'].apply(lambda x: ' '.join(x))
    # Vectorize genres
    count_vectorizer = CountVectorizer()
    genre_matrix = count_vectorizer.fit_transform(movies_df['genres_str'])
    # Compute similarity
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    # Find index of the movie
    movie_idx = movies_df[movies_df['title'] == title].index[0]
    similar_movies = list(enumerate(cosine_sim[movie_idx]))
    # Sort by similarity (movie_index, similar score)
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1: top_n+1]
    # Get movie titles
    recommend_list = [movies_df.iloc[item[0]]['title'] for item in similar_movies]
    recommendations = pd.DataFrame({'Sl.No': range(1, len(recommend_list) + 1), 'Movie Title': recommend_list})
    return recommendations
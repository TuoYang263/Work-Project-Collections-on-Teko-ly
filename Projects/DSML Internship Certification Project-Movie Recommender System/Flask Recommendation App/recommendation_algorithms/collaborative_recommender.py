import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_recommender(movies_df, ratings_df, user_id, top_n, k):
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    # Compute similarity
    user_similarity = cosine_similarity(user_item_matrix)
    # Find similar users (user_similarity[user_idx] is a list, denote the current user's similarity to others)
    user_idx = user_id - 1
    similar_users = sorted(enumerate(user_similarity[user_idx]), key=lambda x: x[1], reverse=True)[:k]
    # Get recommendations
    similar_user_indices = [user[0] for user in similar_users]
    similar_users_ratings = user_item_matrix.iloc[similar_user_indices].mean(axis=0)
    recommendations = similar_users_ratings.sort_values(ascending=False).head(top_n)
    recommended_movie_ids = recommendations.index.tolist()
    recommended_titles = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title'].tolist()
    # Summarize all the information into a dataframe
    recommended_df = pd.DataFrame({'S.No': range(1, len(recommended_titles) + 1), 
                                   'Movie Title': recommended_titles})
    return recommended_df
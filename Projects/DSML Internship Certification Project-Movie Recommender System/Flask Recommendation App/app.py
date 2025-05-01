import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from support_modules.data_preprocessing import load_data, preprocess_data
from recommendation_algorithms.content_based_recommender import content_based_recommender
from recommendation_algorithms.collaborative_recommender import collaborative_recommender
from recommendation_algorithms.popularity_recommender import popularity_recommender

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
movies_df, ratings_df = load_data()
movies_df = preprocess_data(movies_df)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/popularity', methods=['POST'])
def popularity():
    genre = request.form['genre']
    min_reviews = int(request.form['min_reviews'])
    top_n = int(request.form['top_n'])
    recommendations = popularity_recommender(movies_df, ratings_df, genre, min_reviews, top_n)
    return render_template('results.html', recommendations=recommendations.to_dict(orient='records'), recommender_type='popularity')

@app.route('/content', methods=['POST'])
def content():
    title = request.form['title']
    top_n = int(request.form['top_n'])
    recommendations = content_based_recommender(movies_df, title, top_n)
    return render_template('results.html', recommendations=recommendations.to_dict(orient='records'), recommender_type='content')

@app.route('/collaborative', methods=['POST'])
def collaborative():
    user_id = int(request.form['user_id'])
    top_n = int(request.form['top_n'])
    k = int(request.form['k'])
    recommendations = collaborative_recommender(movies_df, ratings_df, user_id, top_n, k)
    return render_template('results.html', recommendations=recommendations.to_dict(orient='records'), recommender_type='collaborative')

if __name__ == '__main__':
    app.run(debug=True)

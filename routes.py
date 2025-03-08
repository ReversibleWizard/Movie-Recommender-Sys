import pandas as pd

from app import app
from get_data import knn_model, mlb, movie_df, get_genre_id, combine_features
from flask import jsonify, request


@app.route('/v1/recommend', methods=['POST'])
def recommend_movies():
    data = request.json
    duration = data.get('duration', None)
    selected_genres_name = data.get('genres', [])
    director = data.get('director', '')
    actor = data.get('actor', '')

    # Convert genres to genre IDs
    selected_genres = [get_genre_id(genre.capitalize()) for genre in selected_genres_name if get_genre_id(genre.capitalize())]

    # Create an input vector matching the KNN model features
    input_vector = []

    # Add duration
    input_vector.append(duration if duration is not None else 0)

    # Handle genres (One-Hot Encoding)
    genre_vector = [1 if genre in selected_genres else 0 for genre in mlb.classes_]
    input_vector.extend(genre_vector)

    # Handle director
    director_vector = [1 if director == dir_name else 0 for dir_name in movie_df["director"].unique()]
    input_vector.extend(director_vector)

    # Handle actors
    actor_list = set(actor for actors in movie_df["actors"].fillna("").apply(lambda x: x.split(", ")) for actor in actors)
    actor_vector = [1 if actor in actor_list else 0 for actor in actor_list]
    input_vector.extend(actor_vector)

    # Debugging: Print the input vector before trimming
    print("Input Vector Before Trimming:", input_vector)

    # Ensure input vector length matches KNN training
    input_vector = input_vector[:len(knn_model._fit_X[0])]

    # Convert input_vector to DataFrame with the same columns as used in training
    X_train = combine_features(movie_df)  # Ensure we get column names from training
    input_df = pd.DataFrame([input_vector], columns=X_train.columns)


    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(input_df)

    # Get recommended movies
    recommended_movies = movie_df.iloc[indices[0]].to_dict(orient='records')

    # Filter by director and actor
    if director:
        recommended_movies = [movie for movie in recommended_movies if movie['director'] == director]
    if actor:
        recommended_movies = [movie for movie in recommended_movies if actor in movie['actors'].split(", ")]

    print("Generated Input Vector:", input_vector)

    return jsonify(recommended_movies)


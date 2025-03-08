from app import TMDB_API_KEY, TMDB_BASE_URL, requests, pd, MultiLabelBinarizer, NearestNeighbors


# Function to fetch movies from TMDb
def fetch_movies():
    url = f"{TMDB_BASE_URL}/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page=1"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error fetching movies: {response.status_code} - {response.text}")
        return []  # Return an empty list if there's an error

    data = response.json()

    # Check if 'results' is in the response
    if 'results' in data:
        return data['results']
    else:
        print("Unexpected response format:", data)  # Print the unexpected response
        return []  # Return an empty list if 'results' is not found


def fetch_movie_credits(movie_id):
    """Fetches actors and director for a given movie ID."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching credits for movie {movie_id}: {response.status_code}")
        return {"actors": [], "director": "Unknown"}

    data = response.json()

    # Extract director
    director = next((member["name"] for member in data.get("crew", []) if member["job"] == "Director"), "Unknown")

    # Extract top 5 actors
    actors = [member["name"] for member in data.get("cast", [])[:5]]

    return {"actors": actors, "director": director}


# Function to create a DataFrame of movies
def create_movie_dataframe(movies):
    movie_data = []
    for Movie in movies:
        movie_data.append({
            'id': Movie['id'],
            'title': Movie['title'],
            'genre_ids': Movie['genre_ids'],  # Keep this as a list
            'duration': Movie.get('runtime', 0)  # Runtime in minutes
        })
    return pd.DataFrame(movie_data)

def create_genre_dataframe():
    url = f"{TMDB_BASE_URL}/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)

    if response.status_code == 200:
        # Extract genres from the response
        genres = response.json()['genres']
        # Create a DataFrame from the genres list
        genreDf = pd.DataFrame(genres)
        return genreDf
    else:
        print(f"Error fetching genres: {response.status_code} - {response.text}")
        return pd.DataFrame(columns=['id', 'name'])  # Return an empty DataFrame with the correct columns

# Load movies and create DataFrame
movie = fetch_movies()
if not movie:
    print("No movies fetched. Exiting.")
    exit(1)  # Exit if no movies are fetched

movie_df = create_movie_dataframe(movie)
genre_df = create_genre_dataframe()

movie_df["actors"] = None
movie_df["director"] = None

for idx, row in movie_df.iterrows():
    credits = fetch_movie_credits(row["id"])
    movie_df.at[idx, "actors"] = ", ".join(credits["actors"])  # Store actors as a comma-separated string
    movie_df.at[idx, "director"] = credits["director"]


# Function to get genre ID by name
def get_genre_id(genre_name):
    global genre_df
    # Assuming genre_df is a DataFrame with columns 'name' and 'id'
    genre_row = genre_df[genre_df['name'].str.lower() == genre_name.lower()]
    if not genre_row.empty:
        return genre_row['id'].values[0]  # Return the first matching genre ID
    return None


# Preprocess genres for KNN
mlb = MultiLabelBinarizer()

# Ensure genre_ids is a list of lists
genre_matrix = mlb.fit_transform(movie_df['genre_ids'].apply(lambda x: x if isinstance(x, list) else [x]))


# Combine features for KNN
def combine_features(movies_df):
    # One-hot encode genres
    genre_matrix = mlb.fit_transform(movies_df['genre_ids'].apply(lambda x: x if isinstance(x, list) else [x]))

    # One-hot encode directors
    unique_directors = movies_df["director"].fillna("Unknown").unique()
    director_encoding = {dir_name: idx for idx, dir_name in enumerate(unique_directors)}
    director_vector = movies_df["director"].map(director_encoding).fillna(0)

    # One-hot encode actors
    unique_actors = set(actor for actors in movies_df["actors"].fillna("").apply(lambda x: x.split(", ")) for actor in actors)
    actor_encoding = {actor: idx for idx, actor in enumerate(unique_actors)}
    actor_vector = movies_df["actors"].apply(lambda actors: [actor_encoding[a] for a in actors.split(", ") if a in actor_encoding])

    # Convert actor lists to fixed-size feature vectors (e.g., sum indices)
    actor_vector = actor_vector.apply(lambda x: sum(x) if x else 0)

    # Combine features into a single DataFrame
    combined_df = pd.concat([movies_df[['duration']], pd.DataFrame(genre_matrix)], axis=1)
    combined_df["director"] = director_vector
    combined_df["actors"] = actor_vector

    # Ensure all column names are strings
    combined_df.columns = combined_df.columns.astype(str)

    return combined_df


# KNN Model
def train_knn_model():
    X = combine_features(movie_df)
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X)
    return knn


knn_model = train_knn_model()

# print(movie_df.head())
# print(genre_df.head())

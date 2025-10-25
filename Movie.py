import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os

print("✅ Code is running...")

# 1. Load and Clean Data
file_path = "tmdb_5000_movies.csv"

if not os.path.exists(file_path):
    print(f"❌ Error: Could not find {file_path}")
    print("👉 Make sure 'tmdb_5000_movies.csv' is in the same folder.")
    print("Download from: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata")
    exit()

movies = pd.read_csv(file_path)
movies = movies[['title', 'overview', 'genres', 'keywords']]
movies['overview'] = movies['overview'].fillna('')

def clean_data(x):
    if isinstance(x, str):
        try:
            items = ast.literal_eval(x)
            return " ".join([i['name'] for i in items])
        except:
            return ''
    else:
        return ''

movies['genres'] = movies['genres'].apply(clean_data)
movies['keywords'] = movies['keywords'].apply(clean_data)
movies['combined'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']

print("✅ Data loaded and processed successfully!")

# 2. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['combined'])
similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Recommendation Function
def recommend(movie_name):
    movie_name = movie_name.strip()
    if movie_name not in movies['title'].values:
        print("❌ Movie not found! Try checking capitalization or spelling.\n")
        return
    
    idx = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_movies = scores[1:6]

    print(f"\n🎬 Because you liked '{movie_name}', you might also like:\n")
    for i in top_movies:
        print("👉", movies.iloc[i[0]].title)
    print()  # Add a blank line after each recommendation list

# 4. Continuous Loop for Multiple Recommendations
while True:
    fav_movie = input("\nEnter your favourite movie (or type 'exit' to quit): ")

    if fav_movie.lower() in ['exit', 'quit', 'stop']:
        print("\n👋 Thanks for using Anaisha's movie recommender! Goodbye!")
        break  # exits the loop

    recommend(fav_movie)



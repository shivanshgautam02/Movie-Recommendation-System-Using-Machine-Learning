import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle



movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv") 

movies = movies.merge(credits,on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]  ## selecting the data features that are mainly use for recommendation 


# Combine relevant text features into a single column
movies['combined_features'] = movies['overview'].fillna('') + ' ' + movies['genres'].fillna('') + ' ' + movies['keywords'].fillna('') + ' ' + movies['cast'].fillna('') + ' ' + movies['crew'].fillna('')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example: Get movie recommendations for a given movie title
recommended_movies = get_recommendations('The Dark Knight Rises')

# Save the trained model to a pickle file
with open('movie_recommendation_model_2.pkl', 'wb') as model_file:
    pickle.dump(tfidf_vectorizer, model_file)
    pickle.dump(cosine_sim, model_file)

# The model is saved as 'movie_recommendation_model.pkl' and can be used in a Streamlit app.







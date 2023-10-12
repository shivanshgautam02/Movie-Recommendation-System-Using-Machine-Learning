import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle  # Import the pickle module


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv") 

movies = movies.merge(credits,on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]  ## selecting the data features that are mainly use for recommendation 


# Ensure that the selected columns are treated as strings
columns_to_combine = ['overview', 'genres', 'keywords', 'cast', 'crew']
for column in columns_to_combine:
    movies[column] = movies[column].fillna('')  # Handle missing values by replacing them with empty strings

# Create a TF-IDF vectorizer to convert text data into numerical vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Combine text features (overview, genres, keywords, cast, and crew) into one column
movies['combined_features'] = movies.apply(
    lambda x: ' '.join(x[column] for column in columns_to_combine),
    axis=1
)

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Compute the cosine similarity between movies based on the TF-IDF vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(movie_title):
    # Get the index of the movie that matches the given title
    movie_index = movies[movies['title'] == movie_title].index[0]

    # Get the pairwise cosine similarity scores for all movies with the given movie
    similar_movies_indices = list(enumerate(cosine_sim[movie_index]))

    # Sort the movies based on similarity scores
    similar_movies_indices = sorted(similar_movies_indices, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies (excluding the movie itself)
    top_similar_movies_indices = [index for index, _ in similar_movies_indices[1:11]]

    # Extract movie titles from their indices
    top_similar_movie_titles = movies['title'].iloc[top_similar_movies_indices]

    return top_similar_movie_titles

# Example usage:
if __name__ == "__main__":
    # Train the model
    model = get_recommendations("The Dark Knight")

    # Save the trained model to a pickle file
    pickle.dump(similarity,open('similarity.pkl','wb'))
    
    with open('movie_recommendation_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print("Model saved to movie_recommendation_model.pkl")






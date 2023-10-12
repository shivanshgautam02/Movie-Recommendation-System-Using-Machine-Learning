import streamlit as st
import pandas as pd
import pickle
import requests


# Streamlit web page
st.set_page_config(page_title='Movie Recommender', page_icon='🎥')
st.title('🍿 Movie Recommender System')
st.markdown("Select a movie from the dropdown and click 'Show Recommendations' to get movie recommendations.")
st.sidebar.title('About')
st.sidebar.info(
    "This Movie Recommender System suggests movies based on your selected movie's similarity.\n"
    "It uses data from The Movie Database (TMDb).\n"
    "Built with Streamlit and Python.\n\n\n"
    "This system is built by Shivansh Gautam."
)


# Load the trained model and cosine similarity matrix from pickle file
with open('movie_recommendation_model_2.pkl', 'rb') as model_file:
    tfidf_vectorizer = pickle.load(model_file)
    cosine_sim = pickle.load(model_file)


# Load the 'movies' DataFrame from a pickle file
with open('mlist.pkl', 'rb') as movies_file:
    movies = pickle.load(movies_file)

# Function to fetch movie poster
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=0d72cd7210467b278387359a8daa1a9c".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Streamlit UI
st.title('Movie Recommendation System')

# Create a list of movie titles from the 'title' column in the 'movies' DataFrame
movie_titles = movies['title'].tolist()

# Movie selection dropdown
selected_movie = st.selectbox(
    "Select a movie:",
    movie_titles,
)


# Get movie recommendations
def get_recommendations(title, movies, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]  # Get the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

if st.button('Show Recommendations'):
    recommended_movies = get_recommendations(selected_movie, movies)
    st.subheader('Recommended Movies:')
    
    # Create a grid layout for displaying recommendations
    cols = st.columns(3)  # Adjust the number of columns as needed
    
    for i, movie in enumerate(recommended_movies):
        movie_id = movies[movies['title'] == movie]['movie_id'].values[0]
        poster_url = fetch_poster(movie_id)
        cols[i % 3].image(poster_url, caption=movie, use_column_width=True, width=150)  # Adjust image size here

# Display the movie poster of the input movie
st.subheader('Movie Poster')
movie_id = movies[movies['title'] == selected_movie]['movie_id'].values[0]
poster_url = fetch_poster(movie_id)
st.image(poster_url, caption=selected_movie, use_column_width=True, width=150)  # Adjust image size here
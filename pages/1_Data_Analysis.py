import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Analysis", page_icon="📊")
st.title("1. Data Analysis")

@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    df = pd.merge(movies, movie_stats, on='movieId', how='inner')
    df['avg_rating'] = df['avg_rating'].round(2)
    return df

try:
    df = load_data()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Search Filters 🔍")
    
    max_ratings = int(df['num_ratings'].max())
    min_ratings = st.sidebar.slider(
        "Minimum number of ratings", 
        min_value=0, max_value=max_ratings, value=10
    )
    
    all_genres = set()
    for genres_string in df['genres'].dropna().str.split('|'):
        all_genres.update(genres_string)
    genres_list = ["All"] + sorted(list(all_genres))
    selected_genre = st.sidebar.selectbox("Select Genre", genres_list)
    
    min_year = int(df['year'].min()) if pd.notna(df['year'].min()) else 1900
    max_year = int(df['year'].max()) if pd.notna(df['year'].max()) else 2026
    years_list = ["All"] + list(range(max_year, min_year - 1, -1))
    selected_year = st.sidebar.selectbox("Select Year", years_list)
    
    # --- APPLY FILTERS ---
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['num_ratings'] >= min_ratings]
    
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, na=False)]
    if selected_year != "All":
        filtered_df = filtered_df[filtered_df['year'] == selected_year]

    # --- TOP 10 DISPLAY ---
    st.subheader("Top 10 Movies")
    
    if filtered_df.empty:
        st.warning("No movies found with these criteria. Try adjusting the filters!")
    else:
        top_10 = filtered_df.sort_values(by=['avg_rating', 'num_ratings'], ascending=[False, False]).head(10)
        display_cols = ['title', 'avg_rating', 'num_ratings', 'genres']
        
        st.dataframe(
            top_10[display_cols].reset_index(drop=True),
            use_container_width=True,
            column_config={
                "title": "Movie Title",
                "avg_rating": "Average Rating ",
                "num_ratings": "Number of Ratings",
                "genres": "Genre"
            }
        )
        
except FileNotFoundError:
    st.error("Error: Data files not found. Make sure 'movies.csv' and 'ratings.csv' are in the 'data' folder.")
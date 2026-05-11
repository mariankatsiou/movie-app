import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Movie Comparison", page_icon="🎬", layout="wide")
st.title("2. Comparison of Two Movies ")

@st.cache_data
def load_comparison_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['rating_year'] = ratings['date'].dt.year
    
    df = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
    return df

try:
    df = load_comparison_data()
    all_movies = sorted(df['title'].unique())
    
    st.info("💡 Tip: Click inside the box and start typing to search for a specific movie!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        movie1 = st.selectbox("Select the first movie:", all_movies, index=0)
    with col2:
        movie2 = st.selectbox("Select the second movie:", all_movies, index=1 if len(all_movies) > 1 else 0)

    if movie1 and movie2:
        st.markdown("---")
        
        df1 = df[df['title'] == movie1]
        df2 = df[df['title'] == movie2]
        
        # --- 1. STATISTICS ---
        st.subheader("Rating Statistics")
        stat_col1, stat_col2 = st.columns(2)
        
        with stat_col1:
            st.markdown(f"### 🔵 {movie1}")
            st.metric("Average Rating", f"{df1['rating'].mean():.2f}")
            st.metric("Number of Ratings", df1['rating'].count())
            st.metric("Standard Deviation", f"{df1['rating'].std():.2f}")
            
        with stat_col2:
            st.markdown(f"### 🔴 {movie2}")
            st.metric("Average Rating", f"{df2['rating'].mean():.2f}")
            st.metric("Number of Ratings", df2['rating'].count())
            st.metric("Standard Deviation", f"{df2['rating'].std():.2f}")

        st.markdown("---")
        df_combined = pd.concat([df1, df2])

        # --- 2. HISTOGRAM ---
        st.subheader("Histogram of Ratings")
        fig_hist = px.histogram(df_combined, x="rating", color="title", barmode="group",
                                title="Distribution of Ratings",
                                labels={"rating": "Rating (Stars)", "count": "Count", "title": "Movie"},
                                color_discrete_sequence=["#1f77b4", "#d62728"])
        st.plotly_chart(fig_hist, use_container_width=True)

        yearly_stats = df_combined.groupby(['title', 'rating_year']).agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()

        # --- 3. AVG RATING PER YEAR ---
        st.subheader("Average Rating per Year")
        fig_avg = px.line(yearly_stats, x="rating_year", y="avg_rating", color="title", markers=True,
                          title="Average Rating Trend Over Time",
                          labels={"rating_year": "Year of Rating", "avg_rating": "Average Rating", "title": "Movie"},
                          color_discrete_sequence=["#1f77b4", "#d62728"])
        fig_avg.update_layout(yaxis=dict(range=[0, 5.5]))
        st.plotly_chart(fig_avg, use_container_width=True)

        # --- 4. NUM RATINGS PER YEAR ---
        st.subheader("Number of Ratings per Year")
        fig_count = px.line(yearly_stats, x="rating_year", y="num_ratings", color="title", markers=True,
                            title="Number of Users Who Rated the Movie Each Year",
                            labels={"rating_year": "Year of Rating", "num_ratings": "Number of Ratings", "title": "Movie"},
                            color_discrete_sequence=["#1f77b4", "#d62728"])
        st.plotly_chart(fig_count, use_container_width=True)

except FileNotFoundError:
    st.error("Error: Data files not found. Make sure 'movies.csv' and 'ratings.csv' are in the 'data' folder.")
import streamlit as st
import pandas as pd
from sklearn.linear_model import RidgeCV

st.set_page_config(page_title="Recommendation System", layout="wide")
st.title("Movie Recommendation System")

# --- DATA LOADING & MATRIX CREATION ---
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    
    # Create the Data Matrix X (Rows: movies, Columns: users)
    X = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    
    # Movie statistics for general display
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    df_movies = pd.merge(movies, movie_stats, on='movieId', how='left')
    
    # Filter df_movies to only include movies that exist in our X matrix
    df_movies = df_movies[df_movies['movieId'].isin(X.index)]
    
    return df_movies, X

try:
    movies_df, X_matrix = load_data()
    
    # --- SESSION STATE SETUP ---
    if 'users_db' not in st.session_state:
        st.session_state['users_db'] = {} 
    if 'user_ratings' not in st.session_state:
        st.session_state['user_ratings'] = {} 
    if 'logged_in_user' not in st.session_state:
        st.session_state['logged_in_user'] = None

    # --- SIDEBAR: AUTHENTICATION & UTILS ---
    st.sidebar.header("User Authentication")
    
    if st.session_state['logged_in_user'] is None:
        auth_mode = st.sidebar.radio("Choose action:", ["Login", "Register"])
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if auth_mode == "Register":
            if st.sidebar.button("Sign Up"):
                if username in st.session_state['users_db']:
                    st.sidebar.error("Username already exists!")
                elif len(username) > 0 and len(password) > 0:
                    st.session_state['users_db'][username] = password
                    st.session_state['user_ratings'][username] = {}
                    st.sidebar.success("Registration successful! Please login.")
                else:
                    st.sidebar.warning("Please enter username and password.")
                    
        elif auth_mode == "Login":
            if st.sidebar.button("Log In"):
                if username in st.session_state['users_db'] and st.session_state['users_db'][username] == password:
                    st.session_state['logged_in_user'] = username
                    st.rerun() 
                else:
                    st.sidebar.error("Invalid username or password.")
                    
        st.info("You must log in to rate movies and get recommendations.")

    else:
        current_user = st.session_state['logged_in_user']
        st.sidebar.success(f"Logged in as: {current_user}")
        
        if st.sidebar.button("Log Out"):
            st.session_state['logged_in_user'] = None
            st.rerun()
            
        st.sidebar.markdown("---")
        # --- RESET BUTTON ---
        if st.sidebar.button("Clear My Ratings"):
            st.session_state['user_ratings'][current_user] = {}
            st.sidebar.warning("All your ratings have been deleted.")
            st.rerun()

    # --- MAIN PAGE: LOGGED IN USER ---
    if st.session_state['logged_in_user'] is not None:
        current_user = st.session_state['logged_in_user']
        
        # --- RATING SYSTEM ---
        st.subheader("Rate Movies")
        
        my_ratings = st.session_state['user_ratings'][current_user]
        ratings_count = len(my_ratings)
        
        st.write(f"Movies rated: {ratings_count} / 10")
        progress = min(ratings_count / 10.0, 1.0)
        st.progress(progress)
        
        if ratings_count < 10:
            st.info("Rate at least 10 movies to unlock personalized Machine Learning recommendations.")
        
        unrated_movies = movies_df[~movies_df['movieId'].isin(my_ratings.keys())]
        movie_list = sorted(unrated_movies['title'].unique())
        
        selected_title = st.selectbox("Select a movie to rate:", movie_list)
        
        st.write("Click on the stars to submit your rating:")
        # The key is dynamic so the stars reset when the user selects a new movie
        star_rating = st.feedback("stars", key=f"stars_{selected_title}")
        
        if star_rating is not None:
            # st.feedback returns 0 to 4. We add 1 to make it 1 to 5.
            actual_score = float(star_rating + 1)
            movie_id = movies_df[movies_df['title'] == selected_title]['movieId'].values[0]
            st.session_state['user_ratings'][current_user][movie_id] = actual_score
            st.rerun()

       # --- MACHINE LEARNING RECOMMENDATION LOGIC ---
        if ratings_count >= 10:
            st.markdown("---")
            st.subheader("Machine Learning Recommendations")
            st.write("Predictions calculated based on users with similar preferences.")
            
            # Prepare target vector
            y_me = pd.Series(0.0, index=X_matrix.index)
            for m_id, rating in my_ratings.items():
                if m_id in y_me.index:
                    y_me.at[m_id] = rating
            
            # Training/Testing split
            watched_mask = y_me > 0
            X_train = X_matrix[watched_mask]
            y_train = y_me[watched_mask]
            X_test = X_matrix[~watched_mask]
            
            # ΒΕΛΤΙΩΣΗ 1: RidgeCV (Cross Validation)
            # Δοκιμάζει αυτόματα διάφορα "βάρη" για να βρει το βέλτιστο alpha
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Εμφάνιση της βέλτιστης παραμέτρου (καλό για το report σου!)
            st.caption(f"🧠 Model trained successfully. Best regularization parameter (alpha) found: {model.alpha_}")
            
            # Formatting results
            recommendations = pd.DataFrame({
                'movieId': X_test.index,
                'Predicted_Rating': predictions
            }).merge(movies_df, on='movieId')
            
            top_10 = recommendations.sort_values(by='Predicted_Rating', ascending=False).head(10)
            
            # ΒΕΛΤΙΩΣΗ 2: Explainability (Αγαπημένα Είδη)
            # Βρίσκουμε τα αγαπημένα είδη του χρήστη (από ταινίες που έβαλε >= 4)
            favorite_genres = set()
            for m_id, r in my_ratings.items():
                if r >= 4.0:
                    g_string = movies_df[movies_df['movieId'] == m_id]['genres'].values[0]
                    favorite_genres.update(g_string.split('|'))
            
            # Φτιάχνουμε μια συνάρτηση που ελέγχει πόσα είδη ταιριάζουν
            def get_match_reason(movie_genres):
                movie_g_list = set(movie_genres.split('|'))
                matches = favorite_genres.intersection(movie_g_list)
                if matches:
                    return "Matches your taste in: " + ", ".join(matches)
                return "Based on user similarities"
                
            top_10['Match_Reason'] = top_10['genres'].apply(get_match_reason)
            
            st.dataframe(
                top_10[['title', 'Predicted_Rating', 'genres', 'Match_Reason']].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "title": "Movie Title",
                    "Predicted_Rating": st.column_config.NumberColumn(
                        "Predicted Rating", 
                        format="%.2f"
                    ),
                    "genres": "Genres",
                    "Match_Reason": "Why we recommend it"
                }
            )

        # History Expander
        if ratings_count > 0:
            with st.expander("Show my rating history"):
                history_data = []
                for m_id, r in my_ratings.items():
                    m_title = movies_df[movies_df['movieId'] == m_id]['title'].values[0]
                    history_data.append({"Title": m_title, "My Rating": r})
                st.table(pd.DataFrame(history_data))

except FileNotFoundError:
    st.error("Error: Data files not found. Ensure 'movies.csv' and 'ratings.csv' are in the 'data' folder.")
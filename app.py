import streamlit as st

st.set_page_config(page_title="MovieLens Data Mining", page_icon="🍿", layout="wide")

st.title("Data Mining Assignment: MovieLens ")

st.markdown("""
Welcome! This application implements the requirements of the bonus assignment using **Streamlit**.

Use the sidebar menu on the left to navigate through the different pages:
* **1. Data Analysis:** Find top-rated movies using multiple interactive filters (year, genre, minimum ratings).
* **2. Movie Comparison:** Statistical and visual comparison between two selected movies.
* **3. Recommendation System:** User registration/login and a personalized movie recommendation system.
""")

st.info("Please select a category from the sidebar to get started.")
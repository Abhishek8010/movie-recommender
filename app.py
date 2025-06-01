import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model, le_genre, le_industry = pickle.load(f)

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 3em;
        font-weight: bold;
        color: #1f4e79;
    }
    .subtitle {
        text-align: center;
        font-size: 1.5em;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎥 Movie Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Tell us what you like, and we’ll recommend a movie just for you!</div>", unsafe_allow_html=True)
st.markdown("---")

# Input fields
col1, col2, col3 = st.columns(3)

with col1:
    genre = st.selectbox("Select Genre", le_genre.classes_)

with col2:
    duration = st.slider("Select Duration (in hours)", 1.0, 4.0, step=0.1)

with col3:
    rating = st.slider("Preferred Rating", 1.0, 10.0, step=0.1)

col4, col5 = st.columns(2)

with col4:
    release_year = st.selectbox("Select Release Year", list(range(1990, 2024))[::-1])

with col5:
    industry = st.selectbox("Select Industry", le_industry.classes_)

if st.button("🎯 Recommend Me a Movie"):
    input_data = [[
        le_genre.transform([genre])[0],
        duration,
        rating,
        release_year,
        le_industry.transform([industry])[0]
    ]]

    prediction = model.predict(input_data)
    st.success(f"🍿 Recommended Movie: **{prediction[0]}**")

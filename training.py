import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

# Load dataset
df = pd.read_csv('movies_dataset.csv')

# Encode categorical features
le_genre = LabelEncoder()
le_industry = LabelEncoder()

df['genre_encoded'] = le_genre.fit_transform(df['genre'])
df['industry_encoded'] = le_industry.fit_transform(df['industry'])

X = df[['genre_encoded', 'duration', 'rating', 'release_year', 'industry_encoded']]
y = df['movie_name']

# Train a simple KNN classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Save the model and encoders
os.makedirs("model", exist_ok=True)
with open('model.pkl', 'wb') as f:
    pickle.dump((model, le_genre, le_industry), f)

print("Model trained and saved as model.pkl.")

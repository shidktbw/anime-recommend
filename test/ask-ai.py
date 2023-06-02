import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

def recommend_anime(model, user_history):
    unseen_df = df[~df['Title'].isin(user_history)].copy()
    studio = unseen_df['Studio'].values
    producer = unseen_df['Producer'].values
    anime_type = unseen_df['Type'].values
    ratings = model.predict([studio, producer])
    unseen_df.loc[:, 'PredictedRating'] = ratings
    unseen_df = unseen_df.sort_values('PredictedRating', ascending=False)
    recommended_anime = unseen_df.head(5)
    return recommended_anime[['Title', 'Link']].values.tolist()


df = pd.read_csv('train/Anime_data.csv')

label_encoder = LabelEncoder()
df['Studio'] = label_encoder.fit_transform(df['Studio'])
df['Producer'] = label_encoder.fit_transform(df['Producer'])
df['Type'] = label_encoder.fit_transform(df['Type'])

# Загрузка модели
model = load_model('anime.h5')

# История пользователя
user_history = ['Mezzo Forte', 'Mezzo Forte', 'Mezzo Forte', 'Mezzo Forte', 'Mezzo Forte']


recommendations = recommend_anime(model, user_history)
for anime, link in recommendations:
    print(f"Recommended Anime: {anime}\nLink: {link}\n")

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


model = tf.keras.models.load_model('anime.h5')


df = pd.read_csv('train/Anime_data.csv')

# test list anime
my_favorite_anime = ['Marmalade Boy', 'Matantei Loki Ragnarok', 'Ginyuu Mokushiroku Meine Liebe', 
                     'Psychic Academy', 'Versailles no Bara']

favorite_anime_info = df[df['Title'].isin(my_favorite_anime)]


studio_le = LabelEncoder()
producer_le = LabelEncoder()

# LabelEncoder 
studio_le.fit(df['Studio'])
producer_le.fit(df['Producer'])

# Convert data to format
studio_input = studio_le.transform(favorite_anime_info['Studio'].values)
producer_input = producer_le.transform(favorite_anime_info['Producer'].values)

# We obtain the probability distribution from our model
predictions = model.predict([studio_input, producer_input])

# Find the type of anime with the highest probability for each of your favorite anime
recommended_types = np.argmax(predictions, axis=1)

# Use LabelEncoder to convert the predicted types back to their original labels
type_le = LabelEncoder()
type_le.fit(df['Type'])
recommended_types_labels = type_le.inverse_transform(recommended_types)

# Lets find anime of these types in our DataFrame and recommend and input labels
recommended_anime = df[df['Type'].isin(recommended_types_labels)]

print(recommended_anime)

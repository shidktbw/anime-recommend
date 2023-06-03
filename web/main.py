from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your-secret-key'

df = pd.read_csv('train/Anime_data.csv')

label_encoder = LabelEncoder()
df['Studio'] = label_encoder.fit_transform(df['Studio'])
df['Producer'] = label_encoder.fit_transform(df['Producer'])
df['Type'] = label_encoder.fit_transform(df['Type'])

model = load_model('anime.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    session['recommendations'] = []  
    if request.method == 'POST':
        user_history = request.form.getlist('anime')  
        session['recommendations'] = recommend_anime(model, user_history)
    return render_template('index.html', recommendations=session['recommendations'])
    
@app.route('/supported_anime')
def supported_anime():
    return render_template('supported_anime.html')

@app.route('/clear', methods=['GET'])
def clear():
    return render_template('index.html')


# Function for generating recommendations
def recommend_anime(model, user_history):
    unseen_df = df[~df['Title'].isin(user_history)].copy()
    studio = unseen_df['Studio'].values
    producer = unseen_df['Producer'].values
    anime_type = unseen_df['Type'].values
    ratings = model.predict([studio, producer])
    unseen_df.loc[:, 'PredictedRating'] = ratings[:, 0]  
    unseen_df = unseen_df.sort_values('PredictedRating', ascending=False)
    recommended_anime = unseen_df.head(5)
    return recommended_anime[['Title', 'Link']].values.tolist()


if __name__ == '__main__':
    app.run(debug=True)

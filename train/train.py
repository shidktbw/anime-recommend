import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Model


df = pd.read_csv('train/Anime_data.csv')

label_encoder = LabelEncoder()
df['Studio'] = label_encoder.fit_transform(df['Studio'])
df['Producer'] = label_encoder.fit_transform(df['Producer'])
df['Type'] = label_encoder.fit_transform(df['Type'])

train, test = train_test_split(df, test_size=0.2, random_state=42)


studio_input = Input(shape=[1], name="Studio-Input")
producer_input = Input(shape=[1], name="Producer-Input")
type_input = Input(shape=[1], name="Type-Input")


studio_embedding = Embedding(df['Studio'].nunique(), 5, name="Studio-Embedding")(studio_input)
producer_embedding = Embedding(df['Producer'].nunique(), 5, name="Producer-Embedding")(producer_input)
type_embedding = Embedding(df['Type'].nunique(), 5, name="Type-Embedding")(type_input)


x = tf.keras.layers.concatenate([studio_embedding, producer_embedding, type_embedding])

x = Flatten()(x)

x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)

out = Dense(1, activation='relu')(x)


model = Model(inputs=[studio_input, producer_input, type_input], outputs=out)
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit([train['Studio'], train['Producer'], train['Type']], train['Rating'], 
                    batch_size=64, 
                    epochs=100, # early stop = 10 step for this model
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[early_stop])



model.save('anime.h5')


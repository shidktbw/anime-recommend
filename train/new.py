import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

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

# Конкатенация эмбеддингов
x = tf.keras.layers.concatenate([studio_embedding, producer_embedding, type_embedding])

# Добавим сверточные слои
x = Conv1D(32, 3, activation='relu')(x)
x = MaxPooling1D(2)(x) # Изменим размер пула на 2
x = Conv1D(64, 3, activation='relu')(x)
x = MaxPooling1D(2)(x) # Изменим размер пула на 2
x = Conv1D(128, 3, activation='relu')(x)

...


# Глобальный пулинг для перехода к полносвязному слою
x = GlobalMaxPooling1D()(x)

# Добавляем дополнительные полносвязные слои
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

# Выходной слой
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
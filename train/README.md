# Training the Model


The train script contains the code used to train the deep learning model. The model is trained on the Anime_data.csv dataset, with features such as the studio, producer, and type of anime. The model's architecture consists of embedding layers for each feature, concatenated together and passed through a series of fully connected layers. The final output is a single value representing the predicted rating.

The model is saved as anime.h5 and loaded by the application when generating recommendations.

Note that the model takes a while to train, and you might need a powerful machine if you want to train it with a large dataset or for a large number of epochs. The current model is trained for 100 epochs, with early stopping set to stop training if the validation loss does not decrease for 10 consecutive epochs.

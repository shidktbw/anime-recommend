# Anime Recommendation System o(>ω<)o


## Overview

This project is an Anime Recommendation System built using Python, Flask, and a deep learning model developed using TensorFlow. Given a list of anime titles that a user has watched and liked, it will recommend other anime that the user may enjoy <3

## How it works

The application uses a deep learning model trained on an anime dataset (Anime_data.csv). This model predicts the ratings a user might give to anime they haven't seen yet, based on certain features such as the studio that produced the anime, the producer, and the type of anime (TV, movie, etc.). The application then uses these predicted ratings to recommend anime to the user.

## Supported Anime

Here you can see what anime neural network supports: [click](https://github.com/shidktbw/anime-recommend/blob/main/web/templates/supported_anime.html)

## ⚠️ Important Note
Please make sure to enter the exact title of the anime you've watched, as it appears in the supported anime list ([supported anime](https://github.com/shidktbw/anime-recommend/blob/main/web/templates/supported_anime.html)). If the title is not correctly entered, the application will not be able to use it for generating recommendations, and may instead recommend random anime that are not related to your input and also remember to clear recommendations for new generation! 0)0))

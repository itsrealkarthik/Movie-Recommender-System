import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Embedding, Reshape, Concatenate, Dropout, Dense, Activation, Input
from tensorflow.keras import layers
from pprint import pprint
from data_preprocessing import Process
from joblib import dump, load

model = load('model.joblib')

items_dataset = pd.read_csv('Video.csv')
video_dataset = items_dataset[['VideoID','Title','Category','ChannelName']]

ratings_dataset = pd.read_csv('User.csv')
merged_dataset = pd.merge(ratings_dataset, video_dataset, how='inner', on='VideoID')


def recommender_system(user_id, model, n_movies):

  print("")
  print("Movie seen by the User:")
  # Create a new DataFrame with the specified conditions
  new_df = refined_dataset.loc[refined_dataset['UserID'] == user_id, ['Title', 'Category', 'Watch Percentage']]

# Count the number of unique categories
  unique_categories = new_df['Category'].value_counts()
  print(f'Number of unique categories: {unique_categories}')

# Calculate the average watch percentage for each category
  average_watch_percentage = new_df.groupby('Category')['Watch Percentage'].mean()
  print('Average watch percentage for each category:')
  print(average_watch_percentage)

  print("")
  encoded_user_id = user_enc.transform([user_id])
  seen_movies = list(refined_dataset[refined_dataset['UserID'] == user_id]['movie'])
  #random.shuffle(seen_movies)
  unseen_movies = [i for i in range(min(refined_dataset['movie']), max(refined_dataset['movie'])+1) if i not in seen_movies]
  #random.shuffle(unseen_movies)
  model_input = [np.asarray(list(encoded_user_id)*len(unseen_movies)), np.asarray(unseen_movies)]
  predicted_ratings = model.predict(model_input)
  predicted_ratings = np.max(predicted_ratings, axis=1)
  sorted_index = np.argsort(predicted_ratings)[::-1]
  print(sorted_index)
  recommended_movies = item_enc.inverse_transform(sorted_index)
  print("")
  print("RECOMMENDATIONS are:")
  food=0
  history=0
  manufacturing=0
  sci=0
  art=0
  travel=0
  print(recommended_movies[:n_movies])
  recommended_df = refined_dataset[refined_dataset['Title'].isin(recommended_movies[:n_movies])][['Title','Category']]
  category_counts = recommended_df['Category'].value_counts()
  print(category_counts)

print("Enter user id")
user_id= int(input())
print("Enter number of movies to be recommended:")
n_movies = int(input())
recommender_system(user_id,model,n_movies)
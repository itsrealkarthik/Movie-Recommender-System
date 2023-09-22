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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


items_dataset = pd.read_csv('Video.csv')
video_dataset = items_dataset[['VideoID','Title','Category','ChannelName']]

ratings_dataset = pd.read_csv('User.csv')
merged_dataset = pd.merge(ratings_dataset, video_dataset, how='inner', on='VideoID')

W1=1
W2=1
W3=1
W4=1

class Process:
    def run(self):
        def calculate_polarity(review):
            dict=analyzer.polarity_scores(review)
            score=dict['compound']
            if score>=0.3:
                return 1
            elif score<0:
                return 0
            else:
                return -1

        def likedordisliked(row):
            if row['Liked']==1:
                return 1
            elif row['Disliked']==1:
                return -1
            else:
                return 0

        merged_dataset['Sentiment']=merged_dataset['Comment'].apply(calculate_polarity)
        merged_dataset['Liked?']=merged_dataset.apply(likedordisliked, axis=1)

        def calculate_recommendation_score(row):
            # Extract values from the DataFrame row
            watch_percentage = row['WatchPercentage']
            liked_disliked = row['Liked?']
            comment_sentiment = row['Sentiment']
            star_rating = row['StarRating']
            watch_percentage_normalized = watch_percentage / 100.0

            score = (watch_percentage_normalized * W1) + (liked_disliked * W2) + (comment_sentiment * W3) + (star_rating * W4)

            score = min(5, max(0, score))

            return score

        # Apply the calculate_recommendation_score function to the DataFrame
        merged_dataset['Score'] = merged_dataset.apply(calculate_recommendation_score, axis=1)
        refined_dataset = merged_dataset.groupby(by=['UserID','Title','Category','ChannelName'], as_index=False).agg({"Score":"mean"})
        user_enc = LabelEncoder()
        refined_dataset['user'] = user_enc.fit_transform(refined_dataset['UserID'].values)
        item_enc = LabelEncoder()
        refined_dataset['movie'] = item_enc.fit_transform(refined_dataset['Title'].values)
        refined_dataset['rating'] = refined_dataset['Watch Percentage'].values.astype(np.float32)
        return refined_dataset
import os
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

dataset = pd.read_csv('Video/User.csv', encoding='utf')
video_dataset = pd.read_csv('Video/Video.csv')

merged_dataset = pd.merge(dataset, video_dataset, how='inner', on='VideoID')

def calculate_polarity(review):
  dict=analyzer.polarity_scores(review)
  score=dict['compound']
  if score>=0.35:
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

def calculate_recommendation_score(row, W1=0.3, W2=0.4, W3=0.1, W4=0.2):
    # Extract values from the DataFrame row
    watch_percentage = row['WatchPercentage']
    liked_disliked = row['Liked?']
    comment_sentiment = row['Sentiment']
    star_rating = row['StarRating']
    watch_percentage_normalized = watch_percentage / 100.0

    score = (watch_percentage_normalized * W1) + (liked_disliked * W2) + (comment_sentiment * W3) + (star_rating * W4)

    score = min(5, max(0, score))

    return round(score)

# Apply the calculate_recommendation_score function to the DataFrame
merged_dataset['Score'] = merged_dataset.apply(calculate_recommendation_score, axis=1)
avg_highly_rated_movies = merged_dataset.groupby(['Title']).agg({"Score":"mean"})['Score'].sort_values(ascending=False)

def recommendations_location(location):
  x = location
  print("****************************     ****** Location: ", x," ******     ******************************\n")
  location_based_movies = video_dataset[video_dataset['Location'] == x]
  merged_location_movies = pd.merge(merged_dataset, location_based_movies, how='inner', on='VideoID')
  high_rated_movies = merged_location_movies.groupby(['Title']).agg({"Score":"mean"})['Score'].sort_values(ascending=False)
  high_rated_movies = high_rated_movies.to_frame()
  print("These are the top movies that can be naviely suggested to the new users for the requested movie Location:", x, ". Recommendations based on top average ratings.")
  print(high_rated_movies.head(10))
  print("****************************     ******************************     ******************************")
  popular_movies_inlocation = merged_location_movies.groupby(['Title']).agg({"Score":"count"})['Score'].sort_values(ascending=False)
  popular_movies_inlocation = popular_movies_inlocation.to_frame()
  popular_movies_inlocation.reset_index(level=0, inplace=True)
  popular_movies_inlocation.columns = ['Title', 'Number of Users watched']
  print("These are the most popular movies which can be recommended to a new user in",x,"Location. Recommendations based on Popularity")
  print(popular_movies_inlocation.sort_values('Number of Users watched', ascending=False).head(10))
  print("****************************     ******************************     ******************************\n\n\n\n\n")

city= input("Enter City Name: ")
recommendations_location(city)
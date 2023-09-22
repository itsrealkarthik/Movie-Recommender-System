#Import all the required packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import Process

process = Process()
tfidf= process.run()
user_id = int(input("Enter User ID: "))

video = pd.read_csv("Drive/Video.csv")
user=pd.read_csv("Drive/User.csv")

merged_dataset = pd.merge(video, user, how='inner', on='VideoID')
refined_dataset = merged_dataset.groupby(by=['UserID','Title'], as_index=False).agg({"Score":"mean"})

seen_movies = list(refined_dataset[refined_dataset['UserID'] == user_id]['Title'])[:20]

# Find the index of the user movie
similar_movies_list=[]
for i in seen_movies:
    movie_index = video[video['Title'] == i].index[0]
    # Compute the cosine similarities between the user movie and all other movies
    similarity_scores = cosine_similarity(tfidf[movie_index], tfidf)

    similar_movies = list(enumerate(similarity_scores[0]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:20]

    # Define a dictionary to store movie titles and their scores
    movie_scores = {}

    # Iterate through the sorted similar movies
    for i, score in sorted_similar_movies:
        title = video.loc[i, 'Title']
        
        # If the movie title is already in the dictionary, increase its score by a certain number (e.g., 0.1)
        if title in movie_scores:
            movie_scores[title] += 0.3
        else:
            movie_scores[title] = score

    # Convert the dictionary to a list of dictionaries
    for title, score in movie_scores.items():
        similar_movies_list.append({'Title': title, 'Score': score})

    # Create a DataFrame from the list of dictionaries
    similar_movies_df = pd.DataFrame(similar_movies_list)

    # Sort the DataFrame by score in descending order
    similar_movies_df = similar_movies_df.sort_values(by='Score', ascending=False)

    # Reset the index of the DataFrame
    similar_movies_df = similar_movies_df.reset_index(drop=True)
    # Merge the two DataFrames on the 'Title' column
    merged_df = pd.merge(similar_movies_df, df, on='Title', how='left')
merged_df=merged_df[['VideoID','Title','ChannelName','Category','Score']]
merged_df=merged_df.iloc[:50]
print(merged_df)

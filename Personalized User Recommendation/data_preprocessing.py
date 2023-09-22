#Import all the required packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
video = pd.read_csv('Drive/Video.csv')
user=pd.read_csv('Drive/User.csv')
class Process:
    def run(self):
        #Hyper Parameter
        w1=1    #Weight of
        w2=1    #Weight of
        w3=1    #Weight of
        w4=1    #Weight of

        def calculate_recommendation_score(row):
            # Extract values from the DataFrame row
            watch_percentage = row['Watch Percentage']
            liked_disliked = row['Liked?']
            comment_sentiment = row['Sentiment']
            star_rating = row['StarRating']
            watch_percentage_normalized = watch_percentage / 100.0
            score = (watch_percentage_normalized * w1) + (liked_disliked * w2) + (comment_sentiment * w3) + (star_rating * w4)
            score = min(5, max(0, score))
            return score


        merged_dataset = pd.merge(video, user, how='inner', on='VideoID')

        # Apply the calculate_recommendation_score function to the DataFrame
        merged_dataset['Score'] = merged_dataset.apply(calculate_recommendation_score, axis=1)

        refined_dataset = merged_dataset.groupby(by=['UserID','Title'], as_index=False).agg({"Score":"mean"})

        # Combine movie name and tags into a single string
        refined_dataset['content'] = refined_dataset['ChannelName']+refined_dataset['Title'].astype(str).fillna(' ') + ' ' + refined_dataset['Category'].astype(str).fillna(' ') + ' ' + refined_dataset['MainCategories'].fillna(' ') + ' ' + refined_dataset['Description'].fillna(' ')

        # Create bag of words
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(refined_dataset['content'])
        # Convert bag of words to TF-IDF
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(bow)
        # Apply LSA or LSI
        lsa = TruncatedSVD(n_components=50, algorithm='arpack')
        lsa.fit(tfidf)

        return tfidf

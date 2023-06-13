
from app.ml.sentiment.constants import *

import praw
from textblob import TextBlob # import textblob
import numpy as np # import numpy
from rank_bm25 import BM25Okapi # import BM25
import time, datetime

class SentimentAnalysis:
    def __init__(self, team1_name, team2_name, team1_players, team2_players, date_since, date_until, reddit, analysis_type):
        self.team1_name = team1_name
        self.team2_name = team2_name
        self.team1_players = team1_players
        self.team2_players = team2_players
        self.time_since = time.mktime(date_since.timetuple())
        self.time_until = time.mktime(date_until.timetuple())
        self.team1_player_sentiment = []
        self.team2_player_sentiment = []
        self.team1_player_comments = []
        self.team2_player_comments = []
        self.team1_player_combined = []
        self.team2_player_combined = []
        self.team1_positive_results = None 
        self.team2_positive_results = None
        self.team1_negative_results = None
        self.team2_negative_results = None
        self.__reddit = reddit
        self.__basic_analysis_team(1)
        self.__basic_analysis_team(2)
        if analysis_type != 'basic':
            self.__advanced_analysis()

    def get_team_sentiment(self, team_number):
        if team_number == 1:
            return self.team1_player_sentiment
        elif team_number == 2:
            return self.team2_player_sentiment
        
    def get_team_biased_sentiment(self, team_number, context):
        if context == 'Positive':
            return self.team1_positive_results if team_number == 1 else self.team2_positive_results
        elif context == 'Negative':
            return self.team1_negative_results if team_number == 1 else self.team2_negative_results

    @staticmethod
    def __sentiment_element(element):
        return element[1]
    
    @staticmethod 
    def __rank_scores(corpus, terms):
        bm25 = BM25Okapi(corpus)
        reddit_scores = bm25.get_scores(terms)
        return reddit_scores
    
    @staticmethod
    def __rank_top(corpus, terms):
        bm25 = BM25Okapi(corpus)
        top_10_reddits = bm25.get_top_n(terms, corpus, n=10)
        return top_10_reddits
    
    def __basic_analysis_team(self, team_number):
        if team_number == 1:
            team_name = self.team1_name
            team_players = self.team1_players
        elif team_number == 2:
            team_name = self.team2_name
            team_players = self.team2_players

        for player in team_players:
            search_words = [player, team_name]
            reddits = self.__reddit.subreddit('soccer').search(search_words, limit=NUMBER_OF_THREADS) # search for the player and team

            reddit_array = []
            sentiment_array = []
            combined_array = []

            for reddit in reddits:
                reddit_array.append(reddit.title)
                sentiment_array.append(TextBlob(reddit.title).sentiment)

            for i in range(0, len(reddit_array)):
                combined_array.append((reddit_array[i], sentiment_array[i][0]))

            combined_array.sort(key=self.__sentiment_element)

            if team_number == 1:
                self.team1_player_comments.append(reddit_array)
                self.team1_player_combined.append(combined_array)
            elif team_number == 2:
                self.team2_player_comments.append(reddit_array)
                self.team2_player_combined.append(combined_array)

            sentiment_count = 0
            sentiment_total = 0

            for sentiment in sentiment_array:
                if (sentiment[1] >= SUBJECTIVITY_THRESHOLD):
                    sentiment_count += 1
                    sentiment_total += sentiment[0]

            to_add = [player, 0, sentiment_count] if sentiment_total == 0 else [player, sentiment_total / sentiment_count, sentiment_count]
            if team_number == 1:
                self.team1_player_sentiment.append(to_add)
            elif team_number == 2:
                self.team2_player_sentiment.append(to_add)
        
    def __advanced_analysis(self):
        positive_terms = "assist good excellent great recovery amazing fantastic" # search queries, positive terms
        negative_terms = "poor bad miss own awful dissapointment loss " # negative terms

        team1_total_reddits = []
        team2_total_reddits = []

        team1_positive_results = []
        team2_positive_results = []

        team1_negative_results = []
        team2_negative_results = []

        for i in range(0, len(self.team1_player_comments)):
            team1_total_reddits += self.team1_player_comments[i]
            team2_total_reddits += self.team2_player_comments[i]

        total_reddits = team1_total_reddits + team2_total_reddits
        tokenized_reddits = [doc.split(" ") for doc in total_reddits]

        tokenized_query_positive = positive_terms.split(" ")
        tokenized_query_negative = negative_terms.split(" ")

        # positive array
        positive_array = self.__rank_scores(tokenized_reddits,tokenized_query_positive)
        team1_positive_array = positive_array[0:NUMBER_OF_THREADS*TOTAL_PLAYERS] # break into positive array for the two teams for sum
        team2_positive_array = positive_array[NUMBER_OF_THREADS*TOTAL_PLAYERS:len(positive_array)]

        # negative array
        negative_array = self.__rank_scores(tokenized_reddits,tokenized_query_negative)
        team1_negative_array = negative_array[0:NUMBER_OF_THREADS*TOTAL_PLAYERS]  # break into positive array for the two teams for sum
        team2_negative_array = negative_array[NUMBER_OF_THREADS*TOTAL_PLAYERS:len(negative_array)]

        # postive tweets
        team1_positive_results = np.sum(np.reshape(team1_positive_array,(TOTAL_PLAYERS,NUMBER_OF_THREADS)),axis=1) / NUMBER_OF_THREADS
        team2_positive_results = np.sum(np.reshape(team2_positive_array,(TOTAL_PLAYERS,NUMBER_OF_THREADS)),axis=1) / NUMBER_OF_THREADS

        # negative tweets
        team1_negative_results = np.sum(np.reshape(team1_negative_array,(TOTAL_PLAYERS,NUMBER_OF_THREADS)),axis=1) / -NUMBER_OF_THREADS
        team2_negative_results = np.sum(np.reshape(team2_negative_array,(TOTAL_PLAYERS,NUMBER_OF_THREADS)),axis=1) / -NUMBER_OF_THREADS

        ### create bar graphs for Part 2 displaying data and then save down


        # reshape sum arrays for graphing

        self.team1_positive_results = np.round(np.reshape(team1_positive_results,(1,TOTAL_PLAYERS)).tolist(),3)
        self.team2_positive_results = np.round(np.reshape(team2_positive_results,(1,TOTAL_PLAYERS)).tolist(),3)
        self.team1_negative_results = np.round(np.reshape(team1_negative_results,(1,TOTAL_PLAYERS)).tolist(),3)
        self.team2_negative_results = np.round(np.reshape(team2_negative_results,(1,TOTAL_PLAYERS)).tolist(),3)
    
    


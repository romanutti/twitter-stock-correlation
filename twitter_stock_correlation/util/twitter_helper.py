import numpy as np
import pandas as pd
import tweepy

""" 
Imports from application configuration.
.. seealso:: config.py
"""
from util.config import consumerkey
from util.config import consumersecret
from util.config import accesstoken
from util.config import accesstokensecret


class TwitterHelper:
    """
    Class to fetch Twitter data
    """

    def __init__(self):
        """
        Default constructor which initiates object
        """
        pass

    def get_data(self, name, owner):
        """
        Function to refresh Twitter data.

        :param name: The name used as select criteria
                     for tweets
        """
        # Create twitter connection
        self.con = self.get_connection(self)

        # Get companies user
        companyprofile = self.get_user(self, name)

        # Get companies user
        ownerprofile = self.get_user(self, owner)

        # Get tweets
        # Companies tweets
        companytweets = self.get_tweets(self, companyprofile.id_str)
        # Owners tweets
        ownertweets = self.get_tweets(self, ownerprofile.id_str)

        # Enrich data
        # Companies data
        companydata = self.enrich_data(self, companytweets, companyprofile.name)
        # Owners data
        ownerdata = self.enrich_data(self, ownertweets, ownerprofile.name)

        # Combine data
        data = companydata.append(ownerdata)

        self.data = data

    # Twitter connection:
    def get_connection(self):
        """
        Function to create a Twitter connection.

        :return Twitter connection
        """
        # Twitter credentials
        auth = tweepy.OAuthHandler(consumerkey, consumersecret)
        auth.set_access_token(accesstoken, accesstokensecret)

        # Get tweepy API model instance
        con = tweepy.API(auth, wait_on_rate_limit=True)
        return con

    # Twitter user:
    def get_user(self, name):
        """
        Function to get a user id by name.

        :param name: The name used as select criteria
                     for tweets
        :return User for specified name
        """
        # Get user
        user = self.con.get_user(screen_name=name)
        return user

    # Tweets:
    def get_tweets(self, user):
        """
        Function to get tweets for a user.

        :return Unformatted twitter data
        """
        tweets = []
        # Get tweets
        for status in tweepy.Cursor(self.con.user_timeline, user_id=user).items():
            # limit tweets
            tweets.append(status)
        return tweets

    # Convert to dataframe and enrich
    def enrich_data(self, sourcedata, name):
        """
        Function to convert tweets to data frame an enriches with metadata.

        :return Formatted twitter data
        """
        # Replace line breaks in tweet text
        data = pd.DataFrame(
            data=[tweet.text.replace('\n', ' ').replace('\r', '') for tweet in sourcedata],
            columns=['Tweets'])
        # Add metadata
        data['Length'] = np.array([len(tweet.text) for tweet in sourcedata])
        data['Id'] = np.array([tweet.id for tweet in sourcedata])
        data['Date of tweet'] = np.array([tweet.created_at for tweet in sourcedata])
        data['Source'] = np.array([tweet.source for tweet in sourcedata])
        data['Likes'] = np.array([tweet.favorite_count for tweet in sourcedata])
        data['Retweets'] = np.array([tweet.retweet_count for tweet in sourcedata])
        data['User'] = name

        return data

    # Export data to csv
    def export_data(self):
        """
        Function to extract twitter data to csv.
        """

        with open('../data/twitterdata.csv', 'a', encoding='utf-8') as f:
            self.data.to_csv('../data/twitterdata.csv', sep='\t', encoding='utf-8')
        # Header information
        template = "# Tweets regarding \"Tesla\" \n" + \
                   "# ------------------------------------------------------------------- \n" + \
                   "# Formatted export of twitter data on user Tesla or Elon Musk. Besides\n" + \
                   "# the actual text, the dataset contains meta information like length,\n" + \
                   "# date, source, number of likes and number of retweets. The data can be\n" + \
                   "# recreated at any time with the \"load_data.py\"-script. \n" + \
                   "#\n" + \
                   "# Tweets are restricted to user \"Tesla\" or \"Elon Musk\".\n" + \
                   "# To get clean data, line breaks in the tweets were removed.\n" + \
                   "#\n" + \
                   "# Extracted via Twitter API, https://apps.twitter.com/app/15892302 \n" + \
                   "# December, 26, 2018, Marco Romanutti \n" + \
                   "#\n" + \
                   "#\n" + \
                   "{}"""

        with open('../data/twitterdata.csv', 'w', encoding='utf-8') as fp:
            fp.write(template.format(self.data.to_csv(index=False, encoding='utf-8')))


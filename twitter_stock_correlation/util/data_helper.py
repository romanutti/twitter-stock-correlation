import re as re

import dateutil.parser
import pandas as pd
from textblob import TextBlob

""" 
Imports from application configuration.
.. seealso:: config.py
"""
from util.config import yahoodata
from util.config import twitterdata

pd.set_option("display.max_rows", 10)  # display only 10 rows


class DataHelper:
    """
    Class to load and prepare data
    """

    def __init__(self):
        """
        Default constructor which initiates object
        """
        pass

    def prepare_tweets(self):
        """
        Function to load and prepare twitter data.

        :return: Twitter data DataFrame, indexed by date
        """
        # Load twitter data
        raw_tweets = self.load_twitter_data(self)
        # Set DatetimeIndex
        indexed_tweets = self.set_datetimeindex(raw_tweets, 'Date of tweet')
        # Clean tweets
        cleaned_tweets = self.remove_specialcharacters(indexed_tweets)
        # Lowercase tweets
        lowercased_tweets = self.lowercase_text(cleaned_tweets, 'Tweets')
        # Categorize source
        categorized_tweets = self.create_category(lowercased_tweets, 'Source')
        # Rename columns
        renamed_tweets = self.rename_tweet_columns(categorized_tweets)
        # Analyze tweets
        analyzed_tweets = self.get_sentiment_score(renamed_tweets)
        # Drop old data
        new_tweets = self.drop_old_data(analyzed_tweets)

        return new_tweets

    def prepare_stocks(self):
        """
        Function to load and prepare stock data.

        :return: Stock data DataFrame, indexed by date
        """
        # Load yahoo data
        raw_stocks = self.load_stock_data(self)
        # Set DatetimeIndex
        indexed_stocks = self.set_datetimeindex(raw_stocks, 'Date')
        # Calculate daily variation of stock price
        extended_stocks = self.get_daily_difference(indexed_stocks)
        # Drop unused columns, as we want to focus on important ones
        reduced_stocks = self.drop_column(extended_stocks, 'Adjusted closing price (USD)')
        # Round to two decimal places
        rounded_stocks = self.round_data(reduced_stocks)
        # Rename columns
        renamed_stocks = self.rename_stock_columns(rounded_stocks)
        # Drop old data
        new_stocks = self.drop_old_data(renamed_stocks)

        return new_stocks

    def load_stock_data(self):
        """
        Function to load exported stock data.

        :return: Stock data DataFrame, indexed by date
        """
        data = pd.read_csv(filepath_or_buffer='data/' + yahoodata, sep=',', skiprows=15)
        return data

    def remove_specialcharacters(data):
        """
        Function to remove special characters from tweet.

        :param data: Input DataFrame
        :return: DataFrame containing cleaned tweets
        """
        data['Tweets'] = data['Tweets'].map(
            lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
        return data

    def get_sentiment_score(data):
        """
        Function to analyse and score sentiment of a tweet using TextBlob
        (see https://textblob.readthedocs.io/en/dev/api_reference.html)

        :param data: Input DataFrame
        :return: DataFrame containing polarity of tweet
        """
        data['polarity'] = data.tweet.map(lambda x: TextBlob(x).sentiment.polarity)
        return data

    def load_twitter_data(self):
        """
        Function to load exported twitter data.

        :return: Twitter data DataFrame, indexed by date
        """
        data = pd.read_csv(filepath_or_buffer='data/' + twitterdata, sep=',', skiprows=14)
        return data

    def set_index(data, column):
        """
        Function to set an index on the DataFrame.

        :param data: Input DataFrame
        :param column: Name of the column, to be used as index
        :return: DataFrame, indexed by column
        """
        return data.set_index(column)

    def create_category(data, column):
        """
        Function to create categorical data on a given column of the DataFrame.

        :param data: Input DataFrame
        :param column: Name of the column, to be used as category
        :return: DataFrame, specified column replaced by categorical data
        """
        data[column] = data[column].astype('category')
        return data

    def set_datetimeindex(data, column):
        """
        Function to set a DateTimeIndex on the DataFrame.

        :param data: Input DataFrame
        :param column: Name of the column, to be used as DateTimeIndex
        :return: DataFrame, indexed by column
        """
        data[column] = pd.DatetimeIndex(data[column])
        data.set_index(column, inplace=True)
        return data

    def resample_tweets(data):
        """
        Function to resample all tweets by day.

        :param data: Original tweets with a granularity of seconds
        :return: Grouped tweets with indexed and grouped by a granularity of a day
        """
        # Resample Data
        resampled = pd.DataFrame()
        resampled['number_of_tweets'] = data.tweet.resample('D').count()
        resampled['length'] = data.length.resample('D').mean()
        resampled['likes'] = data.likes.resample('D').sum()
        resampled['retweets'] = data.retweets.resample('D').sum()
        resampled['polarity'] = data.polarity.resample('D').mean()

        return resampled

    def rename_tweet_columns(data):
        """
        Function rename tweet DataFrame columns

        :param data: Original tweets
        :return: Standardized column names
        """
        # Rename columns
        data.columns = ['tweet', 'length', 'id', 'source', 'likes', 'retweets', 'user']
        data.index.names = ['date']

        return data

    def rename_stock_columns(data):
        """
        Function rename tweet DataFrame columns

        :param data: Original tweets
        :return: Standardized column names
        """
        # Rename columns
        data.columns = ['highest_price_usd', 'lowest_price_usd', 'opening_price_usd',
                        'closing_price_usd', 'volume', 'daily_diff_usd', 'daily_diff_usd_abs']
        data.index.names = ['date']

        return data

    def drop_old_data(data):
        """
        Function to drop data older than 2018

        :param data: Original DataFrame
        :return: Data > year 2017
        """
        # Drop old data
        data = data[data.index >= dateutil.parser.parse("2018-01-01")]

        return data

    def concatenate_data(twitter, stock):
        """
        Function to concatenate twitter und stock data.

        :param twitter: Twitter data
        :param stock: NASDAQ-100 data
        :return: Combined dataframe, indexed by date
        """
        return pd.concat([stock, twitter], axis=1, sort=True)

    def drop_column(data, column):
            """
            Function to drop unused columns in DataFrame.

            :param data: Original DataFrame
            :param column: Column to be dropped
            :return: DataFrame without dropped column
            """
            return data.drop(columns=column)

    def round_data(data):
        """
        Function to round all columns in DataFrame to two decimal places.

        :param data: Original DataFrame
        :return: DataFrame without rounded columns
        """
        return data.round(2)

    def lowercase_text(data, column):
        """
        Function to lowercase all tweets for an easier analysis.

        :param data: Original DataFrame
        :param column: Column to be lowercased
        :return: DataFrame with lowercased column
        """
        data[column] = data[column].apply(lambda x: x.lower())
        return data

    def get_daily_difference(data):
        """
        Function to calculate the difference between daily opening and closing stock price.

        :param data: Original DataFrame
        :return: DataFrame with difference column
        """
        data['Daily difference (USD)'] = data['Opening price (USD)'] - data['Closing price (USD)']
        data['Daily difference (USD) Absolute'] = data['Daily difference (USD)'].abs()
        return data

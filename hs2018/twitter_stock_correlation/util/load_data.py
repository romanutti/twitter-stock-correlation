"""
The following method acts as an entry-point to the
application.

Example:
    The application can be started as python-script:
        $ python load_data.py
"""
from util.twitter_helper import TwitterHelper as twitterHelper
from util.yahoo_helper import YahooHelper as yahooHelper

""" 
Imports from application configuration.
.. seealso:: config.py
"""
from util.config import symbol
from util.config import name
from util.config import owner


def main():
    """
    Main-method to start application
    """
    # Get Twitter data
    twitterHelper.get_data(twitterHelper, name, owner)
    twitterHelper.export_data(twitterHelper)

    # Get Yahoo data
    yahooHelper.get_data(yahooHelper, symbol)
    yahooHelper.export_data(yahooHelper)


if __name__ == '__main__':
    main()






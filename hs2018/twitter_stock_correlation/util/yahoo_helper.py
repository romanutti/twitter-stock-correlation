from pandas_datareader import data as pdr
from datetime import date


class YahooHelper:
    """
    Class to fetch Yahoo data
    """

    def __init__(self):
        """
        Default constructor which initiates object
        """
        pass

    def get_data(self, symbol):
        """
        Function to collect Twitter data.

        :param symbol: The Symbol used to identify
                       an NASDAQ-100 stock.
        """
        # Collect stock market data
        self.data = self.get_stock_data(symbol)

    # Symbol lookup:
    def get_stock_data(symbol):
        """
        Function to get stock data for current year by ticker symbol.

        :param symbol: The Symbol used to identify
                       an NASDAQ-100 stock.
        :return: Stock data for current year
        """
        # Set current dates
        start = date(date.today().year, 1, 1)  # first of current year
        end = date.today()  # today

        # Get yahoo Yahoo data
        data = pdr.get_data_yahoo(symbol, start=start, end=end)

        # Rename columns
        data.columns = ["Highest price (USD)",
                        "Lowest price (USD)",
                        "Opening price (USD)",
                        "Closing price (USD)",
                        "Volume",
                        "Adjusted closing price (USD)"]

        return data

    # Export data to csv
    def export_data(self):
        """
        Function to extract stock data to csv.
        """
        with open('../data/yahoodata.csv', 'a', encoding='utf-8') as f:
            self.data.to_csv('../data/yahoodata.csv', sep='\t', encoding='utf-8')
        # Header information
        template = "# TSLA Stocks over time \n" + \
                   "# --------------------------------------------------------------------- \n" + \
                   "# Export of stock data of \"Tesla Inc.\" for current year. The dataset\n" + \
                   "# consists of selected key stock exchange figures on a daily basis. \n" + \
                   "# The data can be recreated at any time with the \"load_data.py\"-script.\n" + \
                   "# The data record contains one record sorted per trading day. \n" + \
                   "#\n" + \
                   "# The data is restricted to the NASDAQ symbol \"TSLA\" which represents \n" + \
                   "# the company Tesla Inc. The stock information was limited to the period \n" + \
                   "# from 1st January to the current day of the year. \n" + \
                   "#\n" + \
                   "# Extracted via Yahoo-Finance API, https://pypi.org/project/yahoo-finance/ \n" + \
                   "# December, 26, 2018, Marco Romanutti \n" + \
                   "#\n" + \
                   "#\n" + \
                   "{}"""

        with open('../data/yahoodata.csv', 'w', encoding='utf-8') as fp:
            fp.write(template.format(self.data.to_csv(index=True, encoding='utf-8')))

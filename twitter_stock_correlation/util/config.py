"""
The following attributes are used to configure the
application before running.

Attributes:
    symbol (string): The Symbol used to identify
        an NASDAQ-100 Stock.
    name (string): The name of the company used
        as select criteria for tweets
    owner (string): The name of the company owner
        used as select criteria for tweets
    consumerkey (string): Twitter api consumer key
    consumerkey (string): Twitter api consumer password
    accesstoken (string): Twitter api access token
    accesstokensecret (string): Twitter api secret
    twitterdata (string): Filename of exported tweets
    yahoodata (string): Filename of exported NASDAQ-100 stock data

"""

# Yahoo
symbol = 'TSLA'
yahoodata = 'yahoodata.csv'

# Twitter
name = 'Tesla'
owner = 'elonmusk'
consumerkey = 'ylrgBVHxtnUwbclBJG2CJctOf'
consumersecret = 'hiNfbYemrm6n2li88gg701zd5TqQ7Vhm2cwof16s4Q7Qxv3upP'
accesstoken = '4458248674-FTetmdyYmEezTLZ0oRmvEeIXWajLQ3KyZZw3spu'
accesstokensecret = 'KHYPIQpQSBw6FC52bPghfld3CIhjREopoYO21B2DsOGCJ'
twitterdata = 'twitterdata.csv'

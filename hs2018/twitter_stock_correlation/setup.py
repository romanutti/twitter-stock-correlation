from setuptools import setup

setup(name='twitter_stock_correlation',
      version='0.1',
      description='Correlation NASDAQ-100 stock with Twitter Data',
      url='https://gitlab.fhnw.ch/ML-AT-FHNW/dsp_data_stories',
      author='Marco Romanutti',
      author_email='marco.romanutti@students.fhnw.ch',
      packages=['util'],
      zip_safe=False, install_requires=['pandas', 'matplotlib',
                                        'tweepy', 'numpy',
                                        'pandas_datareader', 'fix_yahoo_finance',
                                        'textblob', 'scipy',
                                        'seaborn', 'altair',
                                        'vega_datasets', 'vega'])

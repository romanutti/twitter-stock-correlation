# Twitter Stock Correlation

Analysis of the correlation between tweets and stock prices, Module *dsp*@FHNW

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To install the required modules, open a command line and run the following command from this modules source directory:

```
$ pip install .
```

### Gather data

The analyses are based on two separate data sets: First, a formatted export of tweets on user Tesla or Elon Musk. And second, an export of stock data of "Tesla Inc." for current year. The data used in the application can be created using the tool in the *util*-directory. The application can be configured via *config.py*. 

To create the data files run the following command:

```
$ python load_data.py
```

### View data stories

The preparation and analysis of the data is described in the notebooks *DataStory.ipynb* and *DataStory_Appendix.ipynb*.

## Authors

* **Marco Romanutti** - [FHNW](https://gitlab.fhnw.ch/ML-AT-FHNW/dsp_data_stories)

## Acknowledgments

* Documentation according to [reStructuredText](http://docutils.sourceforge.net/rst.html) 
* Project structure follows [The Hitchhiker's Guide to Python](https://docs.python-guide.org/writing/structure/) 
* Tweet polarity classified using [TextBlob](https://textblob.readthedocs.io/en/dev/api_reference.html)

import pandas as pd
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns

from altair.expr import datum


class StatHelper:
    """
    Class to generate statistics and plots
    """

    def __init__(self):
        """
        Default constructor which initiates object
        """
        pass

    def describe_data(data):
        """
        Function to retrieve most important statistical metric.

        :param data: Dataframe on which the calculations should take place
        :return: Statistical metrics, rounded to two decimal places
        """
        return data.describe().round(2)

    def describe_types(data):
        """
        Function to retrieve data types of the variables of the DataFrame

        :param data: DataFrame to describe
        :return: Variable description
        """
        # switch axis
        transposed = pd.DataFrame({'Variables': list(data)})
        # add data type
        transposed['Data types'] = transposed['Variables'].map(lambda x: data[x].dtype)

        return transposed

    def get_mean(data, column):
        """
        Function to retrieve the mean of a specified column.

        :param data: Complete DataFrame
        :param column: Column on which the calculations should take place
        :return: Mean
        """
        return data[column].mean()

    def get_max(data, column):
        """
        Function to retrieve the max of a specified column.

        :param data: Complete DataFrame
        :param column: Column on which the calculations should take place
        :return: Max
        """
        return data[column].max()

    def calc_skew(data, mean, std, skew):
        """
        Function to retrieve the max of a specified column.

        :param data: Original DataFrame
        :param mean: Mean
        :param std: Standard deviation
        :param skew: Skewness value
        :return: Skewed DataFrame
        """
        t = (data - mean) / std
        return 2 / std * scipy.stats.norm.pdf(t) * scipy.stats.norm.cdf(skew*t)

    def plot_source_frequency(data):
        """
        Function to plot the number of twets per category.

        :param data: Tweets DataFrame
        """

        # plot number of tweets per category
        diag = data.source.value_counts().plot(kind='bar', title='Kanal der Tweets')
        plt.ylabel('Anzahl tweets')
        plt.xlabel('Source')

        diag.plot()

    def plot_polarity_histogram(data):
        """
        Function the histogram, the normalized histogram
        and the cumulated distribution of polarity.

        :param data: Tweets DataFrame
        """
        # General infos corncering polarity variable
        x = data.polarity
        counts, bins = np.histogram(x, bins=20)
        mean = x.mean()
        std = x.std()
        median = np.median(x)
        quart_l, quart_u = np.percentile(x, [25, 75])
        quant_10, quant_90 = np.percentile(x, [10, 90])

        # Print polarity histogram
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 5))
        _ = ax[0].bar(bins[1:], counts, width=bins[1] - bins[0])
        ymax = ax[0].get_ylim()[1]
        _ = ax[0].vlines(mean, 0, ymax, color='red', label='mean')
        _ = ax[0].vlines(mean - std, 0, ymax, color='red', linestyle='--', label='mean +/-std')
        _ = ax[0].vlines(mean + std, 0, ymax, color='red', linestyle='--')
        _ = ax[0].set_xlabel('Polarität')
        _ = ax[0].set_ylabel('Häufigkeit')
        _ = ax[0].legend()
        _ = ax[0].set_title('Histogramm der Polarität')

        # print normlized polarity histogram
        _ = ax[1].step(bins[1:], counts/counts.sum())
        ymax = ax[1].get_ylim()[1]
        _ = ax[1].vlines(median, 0, ymax, color='green', label='mean')
        _ = ax[1].vlines(quart_l, 0, ymax, color='green', linestyle='--',
                         label='quartile')
        _ = ax[1].vlines(quart_u, 0, ymax, color='green', linestyle='--')
        _ = ax[1].set_xlabel('Polarität')
        _ = ax[1].set_ylabel('Dichte')
        _ = ax[1].legend()
        _ = ax[1].set_title('Normalisiertes Histogramm der Polarität')

        # print cumulated polarity distribution
        _ = ax[2].step(bins[1:], (counts/counts.sum()).cumsum())
        ymax = ax[2].get_ylim()[1]
        xmin, xmax = ax[2].get_xlim()
        # quartiles
        _ = ax[2].vlines(median, 0, ymax, color='green', label='median')
        _ = ax[2].vlines(quart_l, 0, 0.25, color='green', linestyle='--', label='quartiles')
        _ = ax[2].hlines(0.25, xmin, quart_l, color='green', linestyle='--')
        _ = ax[2].vlines(quart_u, 0, 0.75, color='green', linestyle='--')
        _ = ax[2].hlines(0.75, xmin, quart_u, color='green', linestyle='--')
        # quantiles
        _ = ax[2].vlines(quant_10, 0, 0.1, color='lightgreen', linestyle='--', label='quantiles')
        _ = ax[2].hlines(0.1, xmin, quant_10, color='lightgreen', linestyle='--')
        _ = ax[2].vlines(quant_90, 0, 0.9, color='lightgreen', linestyle='--')
        _ = ax[2].hlines(0.9, xmin, quant_90, color='lightgreen', linestyle='--')
        _ = ax[2].set_title('Kumulative Verteilung der Polarität')
        _ = ax[2].set_xlabel('Polarität')
        _ = ax[2].set_ylabel('Anteil')
        _ = ax[2].legend()

    def plot_polarity_development(data):
        """
        Function to plot the development of the polarity.

        :param data: Combined data DataFrame
        """
        # Prepare data for altair plotting
        source = data
        source['date'] = source.index

        # Plot chart
        chart = alt.Chart(source, title="Entwicklung der Polarität").mark_bar().encode(
            x=alt.X('date:T', title='Datum'),
            y=alt.Y('polarity:Q', title='Polarität'),
            color=alt.condition(
                alt.datum.polarity > 0,
                alt.value("#4879D0"),  # The positive color
                alt.value("#D95E61")  # The negative color
            )
        ).properties(width=600)

        chart.display()

    def plot_stock_development(data):
        """
        Function to plot the development of the stock per day.
        The chart shows the min/max and the opening/closing price
        per day in a boxplot.

        :param data: Stocks DataFrame
        """
        # Prepare data for altair plotting
        # Exclude non-trading days
        source = data[data.opening_price_usd.notnull()]
        # Create date column
        source['date'] = source.index

        # Specify color values
        open_close_color = alt.condition("datum.opening_price_usd < datum.closing_price_usd",
                                         alt.value("#319575"),
                                         alt.value("#ae2813"))

        # Create boxplot border rules
        rule = alt.Chart(source, width=800).mark_rule().encode(
            alt.X(
                'yearmonthdate(date):T',
                scale=alt.Scale(domain=[{"month": 12, "date": 31, "year": 2018},
                                        {"month": 1, "date": 1, "year": 2018}]),
                axis=alt.Axis(format='%m/%d', title='Datum')
            ),
            alt.Y(
                'lowest_price_usd',
                scale=alt.Scale(zero=False),
                axis=alt.Axis(title='Aktienpreis (USD)')
            ),
            alt.Y2('highest_price_usd'),
            color=open_close_color
        )

        # Bind displayed window of both plots
        brush = alt.selection(type='interval', encodings=['x'])

        # Create upper plot
        upper = alt.Chart(source).mark_bar().encode(
            alt.X('date:T', scale={'domain': brush.ref()}),
            y='opening_price_usd',
            y2='closing_price_usd',
            color=open_close_color,
            tooltip=['highest_price_usd', 'lowest_price_usd', 'opening_price_usd', 'closing_price_usd']
        ).properties(
            height=400,
            title="Entwicklung des Aktienkurses"
        )
        lower = alt.Chart(source).mark_area().encode(
            alt.X('date:T', scale={'domain': brush.ref()}, title='Monat'),
            alt.Y('closing_price_usd:Q', title='Aktienpreis (USD)'),
        ).properties(
            height=60,
            width=800
        ).add_selection(brush)

        # Concat both charts
        chart = alt.vconcat(rule + upper, lower, data=source)

        # Workaround for interactive charts in presentation mode
        chart.save("resources/stock_development.html")
        # chart.display()

    def plot_stock_volatility(data):
        """
        Function to plot the correlation between daily difference
        and traded volume.

        :param data: Stocks DataFrame
        """
        # Prepare data for altair plotting
        source = data

        chart = alt.Chart(source).mark_circle().encode(
            x=alt.X('daily_diff_usd_abs', axis=alt.Axis(title='Absolute Kursdifferenz (USD)')),
            y=alt.Y('volume', axis=alt.Axis(title='Volumen'))
        ).interactive(
        ).properties(title="Korrelation zwischen Kursdifferenz und Volumen, corr=" +
                           '%.2f' % data.daily_diff_usd_abs.corr(data.volume))

        # Workaround for interactive charts in presentation mode
        chart.save("resources/stock_volatility.html")
        # chart.display()

    def plot_polarity_distribution(self, data):
        """
        Function to plot the distribution of polarity.

        :param data: Twitter DataFrame
        """
        # Prepare data
        source = data
        source.user = source.user.astype(str)
        # Condition is user tesla
        is_user_tesla = source.user.str.contains('Tesla')
        # Condition tweet concerning tesla
        keywords = ['tesla', 'car', 'Tesla', 'Car']
        # Apply conditions
        source = source[source.tweet.str.contains('|'.join(keywords)) | is_user_tesla]

        # Generate boxplot
        chart = self.boxplot(source, x='user', y='polarity')

        chart.display()

    def plot_twitter_activities(data):
        """
        Function to plot the development and ratio of
        twitter activities.

        :param data: Twitter DataFrame
        """
        # Prepare data
        source = data
        # Create date column
        source['date'] = source.index
        # Calculate total twitter activities
        source['total'] = source.likes + source.retweets
        # Unpivot DataFrame
        source = source[['date', 'number_of_tweets', 'likes', 'retweets', 'total']]
        source = source.melt('date', var_name='activities', value_name='count')
        source = source.set_index(['date'])
        # Get week part from date
        source['week'] = source.index.week
        # Delete index
        source = source.reset_index()

        interval = alt.selection_interval(encodings=['x', 'y'])

        # Group data
        base = alt.Chart(source).transform_aggregate(
            total="sum(count)",
            groupby=['week', 'activities']
        )

        # Upper plot
        scatter = base.mark_line().encode(
            alt.X('week:Q', title=''),
            alt.Y('total:Q', title='Total Aktivitäten'),
            color=alt.condition(interval, 'activities:N', alt.value('lightgrey'))
        ).properties(
            selection=interval,
            height=70, width=600,
            title='Entwicklung der Twitter-Aktivitäten'
        ).transform_filter(
            filter=datum.activities == 'total'
        )

        # Lower plot
        bar = base.mark_bar(opacity=0.7).encode(
            alt.X('week:N', title='Woche'),
            alt.Y('total:Q', title='Anzahl Aktivitäten'),
            color=alt.condition(interval, 'activities:N', alt.value('lightgrey')),
        ).properties(height=200, width=600
                     ).transform_filter(
            filter=datum.activities != 'total'
        )

        # Concatenate diagrams
        chart = scatter & bar

        # Workaround for interactive charts in presentation mode
        chart.save("resources/twitter_activities.html")
        # chart.display()

    def plot_diff_correlations(data):
        """
        Function to plot the correlation between
        daily difference and various twitter variables.

        :param data: Combined data DataFrame
        """
        # Prepare data
        source = data
        # Create date column
        source['date'] = source.index
        # Unpivot DataFrame
        source = source[['date', 'daily_diff_usd', 'number_of_tweets', 'likes', 'retweets', 'polarity']]
        source = source.melt(['date', 'daily_diff_usd'], var_name='dependent_variable', value_name='measure')
        # Delete index
        source = source.set_index(['date'])

        alt.Chart(source).mark_point().encode(
            x='measure:Q',
            y='daily_diff_usd:Q',
            color='dependent_variable:N'
        )

        input_dropdown = alt.binding_select(options=['number_of_tweets', 'likes', 'retweets', 'polarity'])
        selection = alt.selection_single(fields=['dependent_variable'], bind=input_dropdown, name='Choose')
        color = alt.condition(
            selection,
            alt.Color('dependent_variable:N', legend=None),
            alt.value('lightgray'))

        chart = alt.Chart(source).mark_point().encode(
            x=alt.X('measure:Q', title='Wert der ausgewählten abh. Variable'),
            y=alt.Y('daily_diff_usd:Q', title='Kursdifferenz (USD)'),
            color='dependent_variable:N',
            tooltip='date:T'
        ).add_selection(
            selection
        ).transform_filter(
            selection
        ).properties(title="Korrelation Kursdifferenz/abhängige Variable")

        # Workaround for interactive charts in presentation mode
        chart.save("resources/diff_correlations.html")
        # chart.display()

    def plot_explorative_correlations(data):
        """
        Function to plot correlations of all variables.

        :param data: Combined data DataFrame
        """
        # Replace NaN values with zeros
        cleansed = data
        cleansed.highest_price_usd.fillna(0, inplace=True)
        cleansed.lowest_price_usd.fillna(0, inplace=True)
        cleansed.opening_price_usd.fillna(0, inplace=True)
        cleansed.closing_price_usd.fillna(0, inplace=True)
        cleansed.volume.fillna(0, inplace=True)
        cleansed.number_of_tweets.fillna(0, inplace=True)
        cleansed.length.fillna(0, inplace=True)
        cleansed.likes.fillna(0, inplace=True)
        cleansed.retweets.fillna(0, inplace=True)
        cleansed.polarity.fillna(0, inplace=True)

        # Exclude non-trading days
        highest_barrier = cleansed.highest_price_usd > 0
        lowest_barrier = cleansed.lowest_price_usd > 0

        cleansed = cleansed[highest_barrier & lowest_barrier]

        # Plot potential correlations
        sns.pairplot(cleansed)

    def plot_correlation_matrix(data):
        """
        Function to plot correlationmatrix of all variables.

        :param data: Combined data DataFrame
        """
        # Replace NaN values with zeros
        cleansed = data
        cleansed.highest_price_usd.fillna(0, inplace=True)
        cleansed.lowest_price_usd.fillna(0, inplace=True)
        cleansed.opening_price_usd.fillna(0, inplace=True)
        cleansed.closing_price_usd.fillna(0, inplace=True)
        cleansed.volume.fillna(0, inplace=True)
        cleansed.number_of_tweets.fillna(0, inplace=True)
        cleansed.length.fillna(0, inplace=True)
        cleansed.likes.fillna(0, inplace=True)
        cleansed.retweets.fillna(0, inplace=True)
        cleansed.polarity.fillna(0, inplace=True)

        # Exclude non-trading days
        highest_barrier = cleansed.highest_price_usd > 0
        lowest_barrier = cleansed.lowest_price_usd > 0

        cleansed = cleansed[highest_barrier & lowest_barrier]

        # Create correlation matrix
        corr = cleansed.corr()
        corr.style.background_gradient().set_precision(2)

        # Plot heatmap
        ax = plt.axes()

        sns.heatmap(
            corr,
            cmap="Blues",
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            ax=ax)

        ax.set_title('Korrelations-Matrix')
        plt.show()

    def boxplot(data, x, y, ytype='Q', xtype='N'):
        """
        Function to create a boxplot diagram.

        :param data: Original DataFrame
        :param x: Value on x axis
        :param y: Value on y axis
        :param xtype: Type of x
        :param ytype: Type of y
        :return: Boxplot diagram
        """
        # Create aggregations
        min_agg = f'min({y}):{ytype}'
        max_agg = f'max({y}):{ytype}'
        median_agg = f'median({y}):{ytype}'
        q1_agg = f'q1({y}):{ytype}'
        q3_agg = f'q3({y}):{ytype}'
        x_val = f'{x}:{xtype}'

        # Create base chart
        base = alt.Chart().encode(
            x=alt.X(x_val, axis=alt.Axis(title='User'))
        ).properties(
            width=250,
            title="Verteilung der Polarität"
        )

        # Specify layer 1: Low whisker
        whisker_low = base.mark_rule().encode(
            y=alt.Y(min_agg, axis=alt.Axis(title='Polarität')),
            y2=q1_agg
        )

        # Specify layer 2: Bar
        box = base.mark_bar().encode(
            y=q1_agg,
            y2=q3_agg
        )

        # Specify layer 3: Tick
        midline = base.mark_tick(
            color='white',
            size=120
        ).encode(
            y=median_agg,
            y2=median_agg
        )

        # Specify layer 4: High whisker
        whisker_high = base.mark_rule().encode(
            y=max_agg,
            y2=q3_agg
        )

        # Combine layers
        return alt.layer(whisker_low, box, whisker_high, midline, data=data)


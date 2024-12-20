import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mpld3
import plotly.graph_objects as go

from plotly.subplots import make_subplots


def plot_distribution_rating_abv(ratings_df, beer_df, labels, save = False, interactive = False, label_list=[2,0,1]):
    '''
    Plots the distribution of alcohol by volume (abv) wit the number/proportion of beers in each class

    Parameters :
    - ratings_df: Dataframe with the ratings of the different users
    - beer_df: Dataframe of the beers
    - labels: labels predicted by the GMM
    - save: If we want to save the file
    - interactive: If we want to plot the interactive plot
    - label_list: list to tell us which label corresponds to which class for the 1st GMM
    The list should give the label which corresponds to the following:[universal, neutral, controversial]


    Returns :
    - range_grouped: Various statistics about the beers grouped by their abv (in range)
    '''
    #Deep copy for security
    ratings_df = ratings_df.copy(deep=True)
    ratings_df['labels'] = labels #Apply labels
    merged_df = ratings_df.merge(beer_df, left_index=True, right_on='id').drop(columns = ['id','appearance','aroma','palate','taste','overall'])
    range_grouped = compute_range_grouped(merged_df, labels=label_list)
    if interactive:
        plot_three_interactive(range_grouped, save = save)
    else:
        plot_three(range_grouped, save = save)
    return range_grouped

def compute_range_grouped(merged_df, labels = [2,0,1]):
    '''
    Computes various statistics about the beers grouped by their abv range

    Parameters :
    - merged_df: Merged dataset of the beers abv with the label given to the beer
    - labels: list to tell us which label corresponds to which class for the 1st GMM
    The list should give the label which corresponds to the following:[universal, neutral, controversial]


    Returns :
    - range_grouped: Various statistics about the beers grouped by their abv
    '''
    #Group by abv and compute the various frequencies for each beer
    grouped = merged_df.groupby('abv').agg(total_count=('labels', 'size'),label_match_universal=('labels', lambda x: (x == labels[0]).sum()),label_match_neutral=('labels', lambda x: (x == labels[1]).sum()),label_match_controversial=('labels', lambda x: (x == labels[2]).sum())).reset_index()
    grouped['frequency_universal']=grouped['label_match_universal']/grouped['total_count']
    grouped['frequency_neutral']=grouped['label_match_neutral']/grouped['total_count']
    grouped['frequency_controversial']=grouped['label_match_controversial']/grouped['total_count']

    #Cut to group by abv range
    grouped['abv_range'] = pd.cut(grouped['abv'], bins=100)
    
    #Group by abv range and compute frequencies mean and sum of labeled beers
    range_grouped = grouped.groupby('abv_range').agg(
        avg_abv=('abv', 'mean'),
        avg_frequency_universal=('frequency_universal', 'mean'),
        avg_frequency_neutral=('frequency_neutral', 'mean'),
        avg_frequency_controversial=('frequency_controversial', 'mean'),
        total_beers=('total_count', 'sum'),
        total_beers_universal = ('label_match_universal','sum'),
        total_beers_neutral = ('label_match_neutral','sum'),
        total_beers_controversial = ('label_match_controversial','sum')
    ).dropna().reset_index()
    
    return range_grouped
def plot_three(range_grouped, save=False):
    '''
    Makes a plot of the abv range with the various statistics computed previously

    Parameters:
    - range_grouped: DataFrame with the abv range and their statistics
    - save: if we save or not the generated image under HTML format
    '''
    # Convert avg_abv to numeric if needed
    range_grouped['avg_abv'] = pd.to_numeric(range_grouped['avg_abv'], errors='coerce')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Line plots for the frequencies
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_neutral', marker='o', ax=axes[0], label='Neutral Frequency', color='blue')
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_controversial', marker='o', ax=axes[0], label='Controversial Frequency', color='red')
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_universal', marker='o', ax=axes[0], label='Universal Frequency', color='green')
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Frequency of Label Matches by ABV")
    axes[0].legend()
    axes[0].grid()

    # Subplot 2: Stacked bar plot for total beers by category
    axes[1].bar(range_grouped['avg_abv'], range_grouped['total_beers_neutral'], label='Neutral Beers', color='blue', alpha=0.7, width=0.5)
    axes[1].bar(
        range_grouped['avg_abv'],
        range_grouped['total_beers_controversial'],
        bottom=range_grouped['total_beers_neutral'],
        label='Controversial Beers',
        color='red',
        alpha=0.7,
        width=0.5
    )
    axes[1].bar(
        range_grouped['avg_abv'],
        range_grouped['total_beers_universal'],
        bottom=range_grouped['total_beers_neutral'] + range_grouped['total_beers_controversial'],
        label='Universal Beers',
        color='green',
        alpha=0.7,
        width=0.5
    )

    axes[1].set_xlabel("ABV (Average)")
    axes[1].set_ylabel("Total Beers")
    axes[1].set_title("Total Beers by Category and ABV (Stacked)")
    axes[1].legend()
    axes[1].grid()

    # Set x-axis limits
    axes[0].set_xlim(0, 12)
    axes[1].set_xlim(0, 12)

    plt.tight_layout()
    plt.show()

    # If we want to save or not the file as HTML format
    if save:
        html_path = "aabv_plot.html"
        mpld3.save_html(fig, html_path)


def plot_threes(range_grouped, save = False):
    '''
    Makes a plot of the abv range with th evarious statistics computed previously

    Parameters :
    - range_grouped: Dataframe with the abv range and their statistics
    - save: if we save or not the generated image under HTML format

    '''
    #Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    #Subplot 1: Line plots for the frequencies
    tmp_plot = range_grouped[range_grouped['avg_abv']<12]
    sns.lineplot(data=tmp_plot, x='avg_abv', y='avg_frequency_neutral', marker='o', ax=axes[0], label='Neutral Frequency', color='blue')
    sns.lineplot(data=tmp_plot, x='avg_abv', y='avg_frequency_controversial', marker='o', ax=axes[0], label='Controversial Frequency', color='red')
    sns.lineplot(data=tmp_plot, x='avg_abv', y='avg_frequency_universal', marker='o', ax=axes[0], label='Universal Frequency', color='green')
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Frequency of Label Matches by ABV")
    axes[0].legend()
    axes[0].grid()
    #axes[0].set_xlim(0, 12)


    #Subplot 2: Stacked bar plot for total beers by category
    x = range(len(tmp_plot))
    axes[1].bar(x, tmp_plot['total_beers_neutral'], label='Neutral Beers', color='blue', alpha=0.7)
    axes[1].bar(x, tmp_plot['total_beers_controversial'], bottom=tmp_plot['total_beers_neutral'], label='Controversial Beers', color='red', alpha=0.7)
    axes[1].bar(
        x,
        tmp_plot['total_beers_universal'],
        bottom=tmp_plot['total_beers_neutral'] + tmp_plot['total_beers_controversial'],
        label='Universal Beers',
        color='green',
        alpha=0.7,
    )

    axes[1].set_xlabel("ABV (Average)")
    axes[1].set_ylabel("Total Beers")
    axes[1].set_title("Total Beers by Category and ABV (Stacked)")
    #axes[1].set_yscale("log")  # if want to set y-axis to log scale
    #axes[1].set_xlim(0, 12)
    axes[1].legend()
    axes[1].grid()

    #Add x-axis labels for avg_abv
    step = max(1, int(len(tmp_plot) // 10))
    x_ticks = range(0, len(tmp_plot), step)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels([f"{round(val, 2)}" for val in tmp_plot['avg_abv'][::step]])

    plt.tight_layout()
    plt.show()

    #If we want to save or not the file as HTML format
    if save:
        html_path = "aabv_plot.html"
        mpld3.save_html(fig, html_path)

def plot_three_interactive(range_grouped, save=False):
    '''
    Makes an interactive plot of the abv range with th evarious statistics computed previously

    Parameters :
    - range_grouped: Dataframe with the abv range and their statistics
    - save: if we save or not the generated image under HTML format

    '''
    #Convert to list
    x_values = list(range(len(range_grouped['avg_abv'])))

    #Create subplots: 2 rows and 1 column
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Frequency of Label Matches by ABV", "Total Beers by Category and ABV (Stacked)")
    )

    #Subplot 1: line plot for the frequencies
    fig.add_trace(go.Scatter(
        x=range_grouped['avg_abv'], 
        y=range_grouped['avg_frequency_neutral'],
        mode='lines+markers',
        name='Neutral Frequency',
        marker=dict(color='blue'),
        line=dict(color='blue'),
        hoverinfo='x+y+name'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=range_grouped['avg_abv'], 
        y=range_grouped['avg_frequency_controversial'],
        mode='lines+markers',
        name='Controversial Frequency',
        marker=dict(color='red'),
        line=dict(color='red'),
        hoverinfo='x+y+name'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=range_grouped['avg_abv'], 
        y=range_grouped['avg_frequency_universal'],
        mode='lines+markers',
        name='Universal Frequency',
        marker=dict(color='green'),
        line=dict(color='green'),
        hoverinfo='x+y+name'
    ), row=1, col=1)

    #Subplot 2: Stacked bar plot for the total beers by category
    fig.add_trace(go.Bar(
        x=x_values,
        y=range_grouped['total_beers_neutral'],
        name='Neutral Beers',
        marker=dict(color='blue'),
        hoverinfo='x+y+name'
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=x_values,
        y=range_grouped['total_beers_controversial'],
        name='Controversial Beers',
        marker=dict(color='red'),
        hoverinfo='x+y+name',
        base=range_grouped['total_beers_neutral']
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=x_values,
        y=range_grouped['total_beers_universal'],
        name='Universal Beers',
        marker=dict(color='green'),
        hoverinfo='x+y+name',
        base=range_grouped['total_beers_neutral'] + range_grouped['total_beers_controversial']
    ), row=2, col=1)

    #Update layout for the figure
    fig.update_layout(
        height=800,
        title="Frequency and Total Beers by ABV",
        xaxis_title="ABV (Average)",
        barmode='stack',
        legend_title="Legend",
        hovermode="x unified",
        template="plotly_white"
    )

    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Total Beers", row=2, col=1)

    #Save plot as HTML file
    if save:
        html_path = "aabv_plot.html"
        fig.write_html(html_path)
        print(f"Interactive plot saved as {html_path}")

    fig.show()

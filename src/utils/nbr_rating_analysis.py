import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_distribution_rating_nbr(ratings_df, beer_df, labels, label_list):
    ratings_df = ratings_df.copy(deep=True)
    ratings_df['labels'] = labels
    merged_df = ratings_df.merge(beer_df, left_index=True, right_on='id').drop(columns = ['id','appearance','aroma','palate','taste','overall'])
    grouped_df = plot_three(merged_df,  labels=label_list)
    return grouped_df

def plot_three(merged_df, labels=[0, 1, 2]):
    # Group by 'nbr_ratings' and calculate necessary statistics
    grouped = merged_df.groupby('nbr_ratings').agg(
        total_count=('labels', 'size'),
        label_match_universal=('labels', lambda x: (x == labels[0]).sum()),
        label_match_neutral=('labels', lambda x: (x == labels[1]).sum()),
        label_match_controversial=('labels', lambda x: (x == labels[2]).sum())
    ).reset_index()

    # Calculate frequencies
    grouped['frequency_universal'] = grouped['label_match_universal'] / grouped['total_count']
    grouped['frequency_neutral'] = grouped['label_match_neutral'] / grouped['total_count']
    grouped['frequency_controversial'] = grouped['label_match_controversial'] / grouped['total_count']

    # Bin the 'nbr_ratings' column into ranges and group again
    grouped['nbr_ratings_range'] = pd.cut(grouped['nbr_ratings'], bins=100)
    range_grouped = grouped.groupby('nbr_ratings_range').agg(
        nbr_ratings_mean=('nbr_ratings', 'mean'),
        avg_frequency_universal=('frequency_universal', 'mean'),
        avg_frequency_neutral=('frequency_neutral', 'mean'),
        avg_frequency_controversial=('frequency_controversial', 'mean'),
        total_beers=('total_count', 'sum'),
        total_beers_universal=('label_match_universal', 'sum'),
        total_beers_neutral=('label_match_neutral', 'sum'),
        total_beers_controversial=('label_match_controversial', 'sum')
    ).dropna().reset_index()

    #Figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(data=range_grouped, x='nbr_ratings_mean', y='avg_frequency_neutral', marker='o', ax=axes[0], label='Neutral Frequency', color='blue')
    sns.lineplot(data=range_grouped, x='nbr_ratings_mean', y='avg_frequency_controversial', marker='o', ax=axes[0], label='Controversial Frequency', color='red')
    sns.lineplot(data=range_grouped, x='nbr_ratings_mean', y='avg_frequency_universal', marker='o', ax=axes[0], label='Universal Frequency', color='green')
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Frequency of Label Matches by Number of Ratings")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_xscale("log")


    neutral_mask = range_grouped['total_beers_neutral'] > 0
    controversial_mask = range_grouped['total_beers_controversial'] > 0
    universal_mask = range_grouped['total_beers_universal'] > 0

    sns.lineplot(data=range_grouped[neutral_mask], x='nbr_ratings_mean', y='total_beers_neutral', marker='o', ax=axes[1], label='Neutral Beers', color='blue')
    sns.lineplot(data=range_grouped[controversial_mask], x='nbr_ratings_mean', y='total_beers_controversial', marker='o', ax=axes[1], label='Controversial Beers', color='red')
    sns.lineplot(data=range_grouped[universal_mask], x='nbr_ratings_mean', y='total_beers_universal', marker='o', ax=axes[1], label='Universal Beers', color='green')
    axes[1].set_ylabel("Total Beers")
    axes[1].set_title("Total Beers by Category and Number of Ratings")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid()

    # Adjust x-axis labels for both subplots
    step = max(1, int(len(range_grouped) // 10))  # Adjust step size for readability
    axes[1].set_xticks(range(0, len(range_grouped), step))
    axes[1].set_xticklabels([f"{int(val)}" for val in range_grouped['nbr_ratings_mean'][::step]])
    axes[1].set_xlabel("Number of Ratings (Mean)")
    axes[1].set_xscale("log")

    plt.tight_layout()
    plt.show()

    return grouped

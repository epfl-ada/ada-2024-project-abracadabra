import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_distribution_rating_abv(ratings_df, beer_df, labels, label_of_interest =2):
    ratings_df = ratings_df.copy(deep=True)
    ratings_df['labels'] = labels
    merged_df = ratings_df.merge(beer_df, left_index=True, right_on='id').drop(columns = ['id','appearance','aroma','palate','taste','overall'])
    plot_single(merged_df, label_of_interest)
    range_grouped = plot_three(merged_df)
    return range_grouped

def plot_single(merged_df, label_of_interest):
    grouped = merged_df.groupby('abv').agg(total_count=('labels', 'size'),label_match_count=('labels', lambda x: (x == label_of_interest).sum())).reset_index()
    grouped['frequency']=grouped['label_match_count']/grouped['total_count']

    grouped['abv_range'] = pd.cut(grouped['abv'], bins=100)
    
    # Group by ABV range and aggregate frequencies
    range_grouped = grouped.groupby('abv_range').agg(
        avg_abv=('abv', 'mean'),
        avg_frequency=('frequency', 'mean'),
        total_beers=('total_count', 'sum')  # Sum of total beers for each ABV range
    ).dropna().reset_index()
    
    # Plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot frequency on the primary y-axis
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency', marker='o', ax=ax1, label='Frequency')
    ax1.set_xlabel("ABV (Binned)")
    ax1.set_ylabel("Frequency", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot total beers on the secondary y-axis
    ax2 = ax1.twinx()
    sns.barplot(data=range_grouped, x='avg_abv', y='total_beers', alpha=0.3, ax=ax2, color='gray', label='Total Beers')
    ax2.set_ylabel("Total Beers", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Add titles and legends
    ax1.set_title("Frequency of Label Matches by ABV with Beer Count in Background")
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Reduce the number of x-axis labels (ticks)
    step = max(1, int(len(range_grouped) // 10))  # Adjust the step size to show roughly 10 labels
    ax1.set_xticks(range(0, len(range_grouped), step))
    ax1.set_xticklabels([f"{round(val, 2)}" for val in range_grouped['avg_abv'][::step]])

    plt.tight_layout()
    plt.show()

def plot_three(merged_df, labels = [0,1,2]):
    grouped = merged_df.groupby('abv').agg(total_count=('labels', 'size'),label_match_universal=('labels', lambda x: (x == labels[0]).sum()),label_match_neutral=('labels', lambda x: (x == labels[1]).sum()),label_match_controversial=('labels', lambda x: (x == labels[2]).sum())).reset_index()
    grouped['frequency_universal']=grouped['label_match_universal']/grouped['total_count']
    grouped['frequency_neutral']=grouped['label_match_neutral']/grouped['total_count']
    grouped['frequency_controversial']=grouped['label_match_controversial']/grouped['total_count']

    grouped['abv_range'] = pd.cut(grouped['abv'], bins=100)
    
    # Group by ABV range and aggregate frequencies
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
    
    print(range_grouped.columns)
    # Plot with dual y-axes

    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot frequency on the primary y-axis
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_neutral', marker='o', ax=ax1, label='Neutral Frequency', color='blue')
    ax1.set_xlabel("ABV (Binned)")
    ax1.set_ylabel("Frequency", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_controversial', marker='o', ax=ax1, label='Controversial Frequency', color='red')

    # Plot neutral frequency (in green)
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_universal', marker='o', ax=ax1, label='Universal Frequency', color='green')

    # Plot total beers on the secondary y-axis
    ax2 = ax1.twinx()
    sns.barplot(data=range_grouped, x='avg_abv', y='total_beers', alpha=0.3, ax=ax2, color='gray', label='Total Beers')
    ax2.set_ylabel("Total Beers", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Add titles and legends
    ax1.set_title("Frequency of Label Matches by ABV with Beer Count in Background")
    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Reduce the number of x-axis labels (ticks)
    step = max(1, int(len(range_grouped) // 10))  # Adjust the step size to show roughly 10 labels
    ax1.set_xticks(range(0, len(range_grouped), step))
    ax1.set_xticklabels([f"{round(val, 2)}" for val in range_grouped['avg_abv'][::step]])

    plt.tight_layout()
    plt.show()
    """

# Create the figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Line plots for frequencies
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_neutral', marker='o', ax=axes[0], label='Neutral Frequency', color='blue')
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_controversial', marker='o', ax=axes[0], label='Controversial Frequency', color='red')
    sns.lineplot(data=range_grouped, x='avg_abv', y='avg_frequency_universal', marker='o', ax=axes[0], label='Universal Frequency', color='green')
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Frequency of Label Matches by ABV")
    axes[0].legend()
    axes[0].grid()


    # Subplot 2: Stacked bar plot for total beers by category
    x = range(len(range_grouped['avg_abv']))
    axes[1].bar(x, range_grouped['total_beers_neutral'], label='Neutral Beers', color='blue', alpha=0.7)
    axes[1].bar(x, range_grouped['total_beers_controversial'], bottom=range_grouped['total_beers_neutral'], label='Controversial Beers', color='red', alpha=0.7)
    axes[1].bar(
        x,
        range_grouped['total_beers_universal'],
        bottom=range_grouped['total_beers_neutral'] + range_grouped['total_beers_controversial'],
        label='Universal Beers',
        color='green',
        alpha=0.7,
    )

    axes[1].set_xlabel("ABV (Average)")
    axes[1].set_ylabel("Total Beers")
    axes[1].set_title("Total Beers by Category and ABV (Stacked)")
    #axes[1].set_yscale("log")  # if want to set y-axis to log scale
    axes[1].legend()
    axes[1].grid()

    # Add x-axis labels for avg_abv
    step = max(1, int(len(range_grouped) // 10))  # Adjust step size for readability
    x_ticks = range(0, len(range_grouped), step)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels([f"{round(val, 2)}" for val in range_grouped['avg_abv'][::step]])

    plt.tight_layout()
    plt.show()

    return range_grouped
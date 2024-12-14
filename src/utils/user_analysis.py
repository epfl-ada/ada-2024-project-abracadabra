import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution_grade_per_category(ratings_df, users_df,  attributes = ['appearance', 'aroma', 'palate', 'taste', 'overall']):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns='id')

    print(ratings_df.head())
    for level in rating_user_level:
        ratings_df_level = ratings_df[ratings_df.rating_user_level == level]
        plot_distribution_per_category(ratings_df_level, user_level = level, attributes= attributes)

def plot_distribution_per_category(ratings_df, user_level, attributes):
    ratings_df = ratings_df[attributes]

    sns.violinplot(ratings_df)
    plt.title('Distribution of grades across various attributes for users defined as ' + user_level )
    plt.xlabel("Attributes")
    plt.ylabel("Grade")

    plt.show()

def plot_proportion_controversial_per_category(ratings_df, users_df, beers_df, label, label_list = [2,0,1]):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns=['id','appearance', 'aroma', 'palate', 'taste', 'overall'])
    ratings_df = add_label_to_comment(ratings_df=ratings_df, beers_df=beers_df, labels = label)
    #for level in rating_user_level:
    #    ratings_df_level = ratings_df[ratings_df.rating_user_level == level]
    #    plot_proportion(ratings_df_level, level, label_list = label_list)
    single_plot(ratings_df, label_list = label_list)

def add_label_to_comment(ratings_df, beers_df, labels):
    beers_df_with_label = beers_df.copy()
    beers_df_with_label['label'] = labels 

    ratings_with_labels = ratings_df.merge(beers_df_with_label[['id', 'label']], left_on='id_beer',right_on='id',how='left').drop(columns='id')

    return ratings_with_labels

def plot_proportion(ratings_df, level, label_list = [2,0,1]):
    labels = ['universal','neutral', 'controversial']
    
    label_map = {label_list[i]: labels[i] for i in range(len(labels))}
    label_counts = ratings_df['label'].value_counts().rename(label_map)
    label_frequency = label_counts/label_counts.sum()
    plt.figure(figsize=(10, 6))
    label_frequency.plot(kind='bar', color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    plt.title('Number of Ratings by Label (' + level+')', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('Number of Ratings', fontsize=14)
    plt.show()

def single_plot(ratings_df, label_list = [2,0,1]):
    label_category = ['universal','neutral', 'controversial']

    for i in range(len(label_category)):
        ratings_df.loc[ratings_df['label'] == i, 'label'] = label_category[i]

    enthusiast_proportions = ratings_df[ratings_df['rating_user_level']=='enthusiast']
    connoisseur_proportions = ratings_df[ratings_df['rating_user_level']=='connoisseur']
    novice_proportions = ratings_df[ratings_df['rating_user_level']=='novice']

    enthusiast_proportions = enthusiast_proportions.groupby('label').size()
    connoisseur_proportions = connoisseur_proportions.groupby('label').size()
    novice_proportions = novice_proportions.groupby('label').size()

    enthusiast_proportions = enthusiast_proportions/enthusiast_proportions.sum()
    connoisseur_proportions = connoisseur_proportions/connoisseur_proportions.sum()
    novice_proportions = novice_proportions/novice_proportions.sum()

    proportions_df = pd.DataFrame({
        'enthusiast': enthusiast_proportions,
        'connoisseur': connoisseur_proportions,
        'novice': novice_proportions})

    proportions_df.plot(kind='bar', figsize=(8, 6))

    # Add plot details
    plt.title('Proportion of Labels by Rating User Level')
    plt.ylabel('Proportion')
    plt.xlabel('Label')
    plt.xticks(rotation=0)
    plt.legend(title='Rating User Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
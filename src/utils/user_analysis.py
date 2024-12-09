import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_grade_per_category(ratings_df, users_df,  attributes = ['appearance', 'aroma', 'palate', 'taste', 'overall']):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns='id')

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

def plot_proportion_controversial_per_category(ratings_df, users_df, beers_df, label):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns=['id','appearance', 'aroma', 'palate', 'taste', 'overall'])
    ratings_df = add_label_to_comment(ratings_df=ratings_df, beers_df=beers_df, labels = label)
    for level in rating_user_level:
        ratings_df_level = ratings_df[ratings_df.rating_user_level == level]
        plot_proportion(ratings_df_level, level)
        

def add_label_to_comment(ratings_df, beers_df, labels):
    beers_df_with_label = beers_df.copy()
    beers_df_with_label['label'] = labels 

    ratings_with_labels = ratings_df.merge(beers_df_with_label[['id', 'label']], left_on='id_beer',right_on='id',how='left').drop(columns='id')

    return ratings_with_labels

def plot_proportion(ratings_df, level):
    labels = ['universal','neutral', 'controversial']
    label_counts = ratings_df['label'].value_counts().rename({0: 'universal', 1: 'neutral', 2: 'controversial'})
    label_frequency = label_counts/label_counts.sum()
    plt.figure(figsize=(10, 6))
    label_frequency.plot(kind='bar', color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    plt.title('Number of Ratings by Label (' + level+')', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('Number of Ratings', fontsize=14)
    plt.show()
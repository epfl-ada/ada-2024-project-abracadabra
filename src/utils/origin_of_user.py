import pandas as pd
import matplotlib.pyplot as plt

from src.utils.countries_map import match_countries, draw_map

def plot_distribution_number_ratings_per_country_of_origin_of_user(users_df, ratings_df, N = 100):
    '''
    Plots the distribution of the total number of ratings made for each country
    
    Parameters :
    - user_df: DataFrame containing user data.
    - ratings_df: Dataframe containing the ratings
    - N: the number of countries to show on the bar plot

    '''
    users_df = correct_number_ratings_per_user(users_df,ratings_df)
    users_df = match_countries(users_df)

    #Count toal number of ratings by location, and sort
    users_df = users_df.groupby('location')['nbr_ratings_total'].sum()    
    users_df = users_df.sort_values(ascending=False).head(N)# if want 100 first countries

    users_df.index = users_df.index.where(users_df.index != 'nan', 'Unknown')

    title = 'Top ' +str(N) + ' countries by total ratings'

    #Plot
    plt.figure(figsize=(16, 12))
    users_df.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel('Country')
    plt.ylabel('Total ratings')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_frequency_user_from_country_label(beers_df,users_df, ratings_df, labels, label_to_match = 0, possible_labels = ['controversial','universal', 'neutral']):
    '''
    Plots the frequency of a labeled beer for users from different countries
    
    Parameters :
    - beers_df: Dataframe containing the beer data
    - user_df: DataFrame containing user data.
    - ratings_df: Dataframe containing the ratings
    - label: attributed by the GMM clustering
    - label_to_match: the label to make the plot about
    - possible labels: list to tell us which label corresponds to which class

    '''

    users_df = correct_number_ratings_per_user(users_df,ratings_df)
    users_nbr_labeled_ratings = compute_nbr_controversial_ratings_per_user(beers_df,users_df, ratings_df, labels, label_to_match)
    users_nbr_labeled_ratings = match_countries(users_nbr_labeled_ratings)

    #Compute frequency of rating of label of interest
    users_nbr_labeled_ratings = users_nbr_labeled_ratings.groupby('location').agg({'nbr_ratings_total': 'sum','nbr_ratings_labelled_matched': 'sum'}).reset_index()
    users_nbr_labeled_ratings['frequency'] = users_nbr_labeled_ratings.nbr_ratings_labelled_matched/users_nbr_labeled_ratings.nbr_ratings_total

    #Plot results
    title = 'Proportion of the ratings of a ' + possible_labels[label_to_match] + ' beer by the users country of origin'
    label = 'Proportion per country'
    draw_map(users_nbr_labeled_ratings[['location','frequency']],title = title, label = label)
    return users_df

def plot_origin_mean_users(beers_df,users_df, ratings_df, labels, label_to_match = 0):
    
    """
    à chaque utilisateur ajouter une colonne sur le nombre de bière qu'il a noté qui sont définis comment lael_to_match_
    trouver l'origin des utilisateurs
    => grouper par origine: 

    use id in users and id_user in ratings_df, id_beer for beer in ratings_df
    """
    users_df = correct_number_ratings_per_user(users_df,ratings_df)
    users_nbr_labeled_ratings = compute_nbr_controversial_ratings_per_user(beers_df,users_df, ratings_df, labels, label_to_match)
    users_nbr_labeled_ratings = match_countries(users_nbr_labeled_ratings)
    users_nbr_labeled_ratings['frequency'] = users_nbr_labeled_ratings.nbr_ratings_labelled_matched/users_nbr_labeled_ratings.nbr_ratings_total
    users_nbr_labeled_ratings = users_nbr_labeled_ratings.groupby('location').agg({'nbr_ratings_total': 'mean','frequency': 'mean'}).reset_index()
    draw_map(users_nbr_labeled_ratings[['location','frequency']],'Frequency of the mean of labeled reviews for the origin of the reviewer')

    return users_df

def correct_number_ratings_per_user(users_df,ratings_df):
    '''
    Recomputes the number of ratings made by user to ensure we have the right value
    
    Parameters :
    - user_df: DataFrame containing user data.
    - ratings_df: Dataframe containing the ratings
    '''
    user_label_counts = ratings_df['id_user'].value_counts()

    users_df_corrected = users_df.copy(deep = True)
    users_df_corrected.set_index('id', inplace=True)
    users_df_corrected['nbr_ratings_total'] = users_df_corrected.index.map(user_label_counts).fillna(0).astype(int)
    users_df_corrected.reset_index(inplace=True)

    users_df_corrected['location'] = users_df_corrected['location'].astype(str)

    return users_df_corrected

def compute_nbr_controversial_ratings_per_user(beers_df, users_df, ratings_df, labels, label_to_match):
    '''
    Computes the number of ratings made of a certain class by user
    
    Parameters :
    - beers_df: DataFrame containing beer data
    - user_df: DataFrame containing user data.
    - ratings_df: DataFrame containing the ratings
    - label: attributed by the GMM clustering
    - label_to_match: the label to make the plot about

    '''
    #Deep copy for safety
    beers_df_with_label = beers_df.copy(deep = True)
    beers_df_with_label['label'] = labels 

    users_nbr_labeled_ratings = users_df.copy(deep = True)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched'] = 0

    #Give label on each rating and filter the ones with the label that interests us
    ratings_with_labels = ratings_df.merge(beers_df_with_label[['id', 'label']], left_on='id_beer',right_on='id',how='left')
    matched_ratings = ratings_with_labels[ratings_with_labels['label'] == label_to_match]

    #Compute by user the number of ratings made of a certain label
    user_label_counts = matched_ratings['id_user'].value_counts()
    users_nbr_labeled_ratings.set_index('id', inplace=True)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched'] = users_nbr_labeled_ratings.index.map(user_label_counts).fillna(0).astype(int)
    users_nbr_labeled_ratings.reset_index(inplace=True)

    return users_nbr_labeled_ratings
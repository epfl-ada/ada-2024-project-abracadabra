import pandas as pd
import matplotlib.pyplot as plt
import os

from src.utils.countries_map import match_countries, draw_map
from src.utils.origin_of_user import correct_number_ratings_per_user

def plots_statistics_origin_beer_wrt_user_origin_sum(beers_df, breweries_df, users_df, ratings_df, labels, label_to_match = 0, possible_labels = ['controversiality','universalality', 'neutrality']):
    '''
    Plots the score mentionned in the notebook by country 
    
    Parameters :
    - beers_df: Dataframe containing the beer data
    - breweries_df: Dataframe containing the brewery data
    - users_df: Dataframe containing the user data
    - ratings_df: Dataframe containing the ratings
    - labels: attributed by the GMM clustering
    - label_to_match: the label to make the plot about
    - possible labels: list to tell us which label corresponds to which class

    '''
    #Clean locations
    users_df = match_countries(users_df)
    breweries_df = match_countries(breweries_df)

    #Correct the number of ratings for the users (total number)
    users_df = correct_number_ratings_per_user(users_df,ratings_df)

    #Compute the ratings for country of origin/abroad that are labelled and total for the users
    users_nbr_labeled_ratings = compute_nbr_origin_of_ratings_per_user(beers_df, users_df,breweries_df, ratings_df,labels, label_to_match)

    #Compute the score for controversiality per country by summing over the location of origin
    users_nbr_labeled_ratings = compute_score_for_controversial_sum(users_nbr_labeled_ratings)

    #Draw map
    draw_map(users_nbr_labeled_ratings[['location','score']],column_for_plot = 'score', min_ = -1,title = 'Map of the score of ' + possible_labels[label_to_match] + ' for the users per country', label='Score per country')

def compute_nbr_origin_of_ratings_per_user(beers_df, users_df,breweries_df, ratings_df,labels, label_to_match):
    '''
    Computes the number total number of ratings grouped by country for te country of origin and the foreign country. 
    The total number of ratings matched to the label is also computed
    
    Parameters :
    - beers_df: Dataframe containing the beer data
    - users_df: Dataframe containing the user data
    - breweries_df: Dataframe containing the brewery data
    - ratings_df: Dataframe containing the ratings
    - labels: attributed by the GMM clustering
    - label_to_match: the label to make the plot about

    '''
    #Apply labels and initialize column we want to compute
    beers_df_with_label = beers_df.copy()
    beers_df_with_label['label'] = labels 

    users_nbr_labeled_ratings = users_df.copy()
    users_nbr_labeled_ratings['nbr_ratings_same_country'] = 0
    users_nbr_labeled_ratings['nbr_ratings_other_country'] = 0
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_same'] = 0
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_other'] = 0

    #Matches to directely comapare the origin of the user and of the origin of the beer
    ratings_df = ratings_df.drop(columns= ['appearance','aroma','palate','taste','overall'])
    ratings_breweries = ratings_df.merge(breweries_df[['id', 'location']], left_on='id_brewery', right_on='id').drop(columns='id').rename(columns={'location':'location_brewery'})
    ratings_breweries_users = ratings_breweries.merge(users_df[['id', 'location']], left_on='id_user', right_on='id').drop(columns='id').rename(columns={'location':'location_user'})
    ratings_full = ratings_breweries_users.merge(beers_df_with_label[['id', 'label']], left_on='id_beer', right_on='id').drop(columns=['id']).rename(columns={'label':'label_beer'})

    #Add a column to tell if the rating was from the same country between user and beer and if the label matches
    ratings_full['same_country'] = ratings_full['location_brewery'] == ratings_full['location_user']
    ratings_full['label_matches'] = ratings_full['label_beer'] == label_to_match

    #Compute the statistics of interest
    country_counts_same = ratings_full[ratings_full['same_country']].groupby('id_user').size()#The user has x rating which come from the same country
    country_counts_other = ratings_full[~ratings_full['same_country']].groupby('id_user').size()#Same as the previous one, just different country

    label_counts_same = ratings_full[ratings_full['same_country'] & ratings_full['label_matches']].groupby('id_user').size()#The user has x rating which come from the same country AND are matched
    label_counts_other  = ratings_full[~ratings_full['same_country'] & ratings_full['label_matches']].groupby('id_user').size()#Same as previous just different country

    users_nbr_labeled_ratings['nbr_ratings_same_country'] = users_nbr_labeled_ratings['id'].map(country_counts_same).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_other_country'] = users_nbr_labeled_ratings['id'].map(country_counts_other).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_same'] = users_nbr_labeled_ratings['id'].map(label_counts_same).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_other'] = users_nbr_labeled_ratings['id'].map(label_counts_other).fillna(0).astype(int)

    return users_nbr_labeled_ratings

def compute_score_for_controversial(users_df):
    users_df['frequency_labelled_same_country'] = users_df.nbr_ratings_labelled_matched_same/users_df.nbr_ratings_same_country
    users_df['frequency_labelled_other_country'] = users_df.nbr_ratings_labelled_matched_other/users_df.nbr_ratings_other_country
    users_df['score'] = users_df['frequency_labelled_same_country']-users_df['frequency_labelled_other_country']
    users_df = users_df.groupby('location').agg({'score': 'mean','frequency_labelled_other_country': 'sum'}).reset_index()
    print("Min score", users_df.score.min(),"Max score", users_df.score.max())
    print(users_df.score.describe())
    return users_df


def compute_score_for_controversial_sum(users_df):
    '''
    Computes the score based on the frequencies in user_df 
    
    Parameters :
    - users_df: Dataframe containing the information about the users grouped by location

    '''
    users_df = users_df.groupby('location').agg({'nbr_ratings_labelled_matched_same': 'sum','nbr_ratings_same_country': 'sum','nbr_ratings_labelled_matched_other': 'sum','nbr_ratings_other_country': 'sum'}).reset_index()

    users_df['frequency_labelled_same_country'] = users_df.nbr_ratings_labelled_matched_same/users_df.nbr_ratings_same_country
    users_df['frequency_labelled_other_country'] = users_df.nbr_ratings_labelled_matched_other/users_df.nbr_ratings_other_country
    users_df['score'] = users_df['frequency_labelled_same_country']-users_df['frequency_labelled_other_country']
    print(users_df.score.describe())
    return users_df

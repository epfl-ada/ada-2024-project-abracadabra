import pandas as pd
import matplotlib.pyplot as plt
import os

from src.utils.countries_map import match_countries, draw_map
from src.utils.origin_of_user import correct_number_ratings_per_user

def plots_beer_origin(beers_df, breweries_df, users_df, ratings_df, labels, label_to_match = 0):
    users_df = match_countries(users_df)
    breweries_df = match_countries(breweries_df)
    users_df = correct_number_ratings_per_user(users_df,ratings_df)
    users_nbr_labeled_ratings = compute_nbr_origin_of_ratings_per_user(beers_df, users_df,breweries_df, ratings_df,labels, label_to_match)
    #print(users_nbr_labeled_ratings.head())

def compute_nbr_origin_of_ratings_per_user(beers_df, users_df,breweries_df, ratings_df,labels, label_to_match):
    beers_df_with_label = beers_df.copy()
    beers_df_with_label['label'] = labels 

    users_nbr_labeled_ratings = users_df.copy()
    users_nbr_labeled_ratings['nbr_ratings_same_country'] = 0
    users_nbr_labeled_ratings['nbr_ratings_other_country'] = 0
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_same'] = 0
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_other'] = 0
    users_nbr_labeled_ratings['nbr_ratings_test'] = 0


    #Matches to directely comapare the origin of the user and of the origin of the beer
    ratings_df = ratings_df.drop(columns= ['date','appearance','aroma','palate','taste','overall','rating',  'text','dataset','matched'])
    print(len(ratings_df))
    ratings_breweries = ratings_df.merge(breweries_df[['id', 'location']], left_on='id_brewery', right_on='id', suffixes=('', '_brewery_var')).drop(columns='id_brewery_var').rename(columns={'location':'location_brewery'})
    print(len(ratings_breweries))
    ratings_breweries_users = ratings_breweries.merge(users_df[['id', 'location']], left_on='id_user', right_on='id', suffixes=('', '_user_var')).drop(columns='id_user_var').rename(columns={'location':'location_user'})
    print(len(ratings_breweries_users))
    ratings_full = ratings_breweries_users.merge(beers_df_with_label[['id', 'label']], left_on='id_beer', right_on='id', suffixes=('', '_beer_var')).drop(columns=['id_beer_var','id'])
    print(len(ratings_full))

    ratings_full['same_country'] = ratings_full['location_brewery'] == ratings_full['location_user']
    ratings_full['label_matches'] = ratings_full['label'] == label_to_match

    country_counts_same = ratings_full[ratings_full['same_country']].groupby('id_user').size()#The user has x rating which come from the same country
    country_counts_other = ratings_full[~ratings_full['same_country']].groupby('id_user').size()#Same as the previous one, just different country

    label_counts_same = ratings_full[ratings_full['same_country'] & ratings_full['label_matches']].groupby('id_user').size()#The user has x rating which come from the same country AND are matched
    label_counts_other  = ratings_full[~ratings_full['same_country'] & ratings_full['label_matches']].groupby('id_user').size()#Same as previous just different country

    users_nbr_labeled_ratings['nbr_ratings_same_country'] = users_nbr_labeled_ratings['id'].map(country_counts_same).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_other_country'] = users_nbr_labeled_ratings['id'].map(country_counts_other).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_same'] = users_nbr_labeled_ratings['id'].map(label_counts_same).fillna(0).astype(int)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched_other'] = users_nbr_labeled_ratings['id'].map(label_counts_other).fillna(0).astype(int)

    #JUST FOR TEST gives same result as before 
    user_label_counts = ratings_df['id_user'].value_counts()
    user_label_counts = ratings_df.groupby('id_user').size()
    #user_label_counts = ratings_full.groupby('id_user').size() #gives the different result.
    users_nbr_labeled_ratings['nbr_ratings_test'] = users_nbr_labeled_ratings['id'].map(user_label_counts).fillna(0).astype(int)

    print(ratings_df[~ratings_df['id_brewery'].isin(breweries_df['id'])])  # Ratings with no matching brewery
    print(ratings_df[~ratings_df['id_user'].isin(users_df['id'])])        # Ratings with no matching user
    print(ratings_df[~ratings_df['id_beer'].isin(beers_df_with_label['id'])])  # Ratings with no matching beer
    return users_nbr_labeled_ratings

    """
    for row in ratings_df.iterrows():
        if breweries_df[breweries_df.id==row.id_user].location == users_df[users_df.id==row.id_user].location:
            users_nbr_labeled_ratings.loc[users_nbr_labeled_ratings.id == users_df.id,['nbr_ratings_same_country']] +=1
            if beers_df_with_label[beers_df_with_label.id == row.id_beer].label == label_to_match:
                users_nbr_labeled_ratings.loc[users_nbr_labeled_ratings.id == users_df.id,['nbr_ratings_labelled_matched_same']] +=1
        else:
            users_nbr_labeled_ratings.loc[users_nbr_labeled_ratings.id == users_df.id,['nbr_ratings_other_country']] +=1
            if beers_df_with_label[beers_df_with_label.id == row.id_beer].label == label_to_match:
                users_nbr_labeled_ratings.loc[users_nbr_labeled_ratings.id == users_df.id,['nbr_ratings_labelled_matched_other']] +=1
    """

    #IN CONSTRUCTION DO NOT MODIFY PLS
    ratings_with_labels = ratings_df.merge(beers_df_with_label[['id', 'label']], left_on='id_beer',right_on='id',how='left')

    matched_ratings = ratings_with_labels[ratings_with_labels['label'] == label_to_match]

    user_label_counts = matched_ratings['id_user'].value_counts()

    users_nbr_labeled_ratings.set_index('id', inplace=True)
    users_nbr_labeled_ratings['nbr_ratings_labelled_matched'] = users_nbr_labeled_ratings.index.map(user_label_counts).fillna(0).astype(int)
    users_nbr_labeled_ratings.reset_index(inplace=True)

    return users_nbr_labeled_ratings



import pandas as pd
import plotly.express as px
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os


#pip install geopandas matplotlib




def compute_proportion_label_per_country(breweries_df,beers_df, beers, labels, label_to_match, possible_labels = ['controversial','universal', 'neutral']):
    '''
    Plots the frequency of a labeled beer for the origin of the beer from different countries
    
    Parameters :
    - breweries_df: Dataframe containing the brewery data
    - beers_df: Dataframe containing the beer data
    - beers: Dataframe with the variance of the grades for each beer
    - label: attributed by the GMM clustering
    - label_to_match: the label to make the plot about
    - possible labels: list to tell us which label corresponds to which class

    '''
    #Get brewery id and label for the beers
    beers['Label'] = labels
    beers = beers.merge(beers_df[['id', 'brewery_id']], left_on='id_beer', right_on='id', how='left')

    amount_beers_per_country_labelled = compute_number_beers_per_country(breweries_df, beers, label_to_match)

    title = 'Map of the proportion of beer labelled as ' + possible_labels[label_to_match] + ' per country'
    draw_map(amount_beers_per_country_labelled[['location','frequency']],title = title, label = 'Proportion of beer per country', min_=0)

def compute_number_beers_per_country(breweries_df, beers, label_to_match):
    """
    Computes the number of beers by country for a given label

    Parameters :
    - breweries_df: Dataframe containing the brewery data
    - beers: Dataframe with the variance of the grades for each beer
    - label_to_match: the label to make the plot about
    """
    #Compute number of label of interest by brewery
    beers_labeled = beers[beers['Label'] == label_to_match]
    label_counts = beers_labeled.groupby('brewery_id').size().reset_index(name='amount_of_labels')

    #Compute frequency of the label by location
    breweries_df = breweries_df.merge(label_counts, how='left', left_on='id', right_on='brewery_id')
    breweries_df['amount_of_labels'] = breweries_df['amount_of_labels'].fillna(0).astype(int)
    breweries_df = breweries_df.drop(columns=['brewery_id'])
    breweries_df = match_countries(breweries_df)
    breweries_df = breweries_df.groupby('location').agg({'amount_of_labels': 'sum','nbr_beers': 'sum'}).reset_index()
    breweries_df['frequency'] = breweries_df.amount_of_labels/breweries_df.nbr_beers
    return breweries_df

def draw_map(countries_Beers_labelled,title, column_for_plot = 'frequency', min_ = 0, label = "Proportion of Beers per Country"):
    '''
    Draw world map with frequency that interests us
    
    Parameters :
    - countries_Beers_labelled: Dataframe with location and data to plot
    - title: title of the plot
    - column_for_plot: which column to use from countries_Beers_labelled
    - min_: min value of the plot for the oclor bar
    - label: label of the plot

    '''
    current_directory = os.getcwd()
    shapefile_path = current_directory+ "/src/utils/data/ne_110m_admin_0_countries.shp"

    #Load world map
    world = gpd.read_file(shapefile_path)
    world = world.merge(countries_Beers_labelled, how='left', left_on='NAME', right_on='location')
    world[column_for_plot] = world[column_for_plot].fillna(0)

    #2 cases: plot between 0 :1 and -1:1. 
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    if min_ ==0:
        world.plot(column=column_for_plot, cmap='OrRd', legend=True,legend_kwds={'label': label},missing_kwds={'color': 'lightgrey', 'label': 'No Data'},ax=ax, vmin = min_, vmax = 1)
    if min_ == -1:
        world.plot(column=column_for_plot, cmap='coolwarm', legend=True,legend_kwds={'label': label},missing_kwds={'color': 'lightgrey', 'label': 'No Data'},ax=ax, vmin = min_,vmax = 1)
    ax.set_title(title, fontsize=16)
    ax.set_axis_off()
    plt.show()

    #Uncomment if want information about te countries missing or not matched
    #unmatched_locations = countries_Beers_labelled[~countries_Beers_labelled['location'].isin(world['NAME'])]
    #unmatched_locations.to_csv('unmatched_locations.txt', index=False, sep='\t')
    #world['NAME'].to_csv('Countries on Map.txt', index=False, sep='\t')

def match_countries(breweries_df):
    '''
    Corrects the name of the countries to get the names needed for the plot to have the most accurate countries
    
    Parameters :
    - breweries_df: DataFrame to correct the location from

    '''
    countries = ["United States", "Canada", "United Kingdom", "Australia", "Germany", "Italy"]

    breweries_df['location'] = breweries_df['location'].fillna('Unknown')

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United States of America' if 'United States' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United States of America' if 'Utah' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United States of America' if 'Hawaii' in x else x)

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United Kingdom' if 'United Kingdom' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United Kingdom' if 'Wales' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United Kingdom' if 'England' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United Kingdom' if 'Scotland' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United Kingdom' if 'Gibraltar' in x else x)

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'Canada' if 'Canada' in x else x)

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'Czechia' if 'Czech Republic' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'Bosnia and Herz.' if 'Bosnia and Herzegovina' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'S. Sudan' if 'South Sudan' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'China' if 'Tibet' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'Slovakia' if 'Slovak Republic' in x else x)

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'Unknown' if 'nan' in x else x)

    return breweries_df

def plot_distribution_number_ratings_per_country_of_origin_of_beer(beers_df, breweries_df_new, ratings_df, N = 100):
    '''
    Plots the distribution of the total number of ratings made for each country
    
    Parameters :
    - beers_df: DataFrame containing beer data.
    - breweries_df_new: DataFrame containing beer data.
    - ratings_df: Dataframe containing the ratings
    - N: the number of countries to show on the bar plot

    '''
    #Merge on beers (deepcopied) brewery location to know where they come from
    beers_df = beers_df.copy(deep = True)
    beers_df = beers_df.merge(breweries_df_new[['id', 'location']], left_on='brewery_id', right_on='id', how='left',suffixes=('', '_breweries')).drop(columns = 'id_breweries')
    
    #Compute the ratings per beer by location
    beers_df = correct_number_ratings_per_beers(beers_df,ratings_df)
    beers_df = match_countries(beers_df)
    beers_df = beers_df.groupby('location')['nbr_ratings'].sum()
    
    beers_df = beers_df.sort_values(ascending=False).head(N)# if want 100 first coountires

    beers_df.index = beers_df.index.where(beers_df.index != 'nan', 'Unknown')

    title = 'Top ' +str(N) + ' Countries by Total Beers'

    #plot
    plt.figure(figsize=(16, 12))
    beers_df.plot(kind='bar', color='skyblue', edgecolor='black', width=0.8)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.xlabel('Country')
    plt.ylabel('Total Beers')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def correct_number_ratings_per_beers(beers_df,ratings_df):
    '''
    Recomputes the correct number of ratings per beer just as a final check
    
    Parameters :
    - beers_df: DataFrame containing beer data.
    - ratings_df: Dataframe containing the ratings

    '''
    beer_label_counts = ratings_df['id_beer'].value_counts()

    beers_df_corrected = beers_df.copy(deep = True)
    beers_df_corrected.set_index('id', inplace=True)
    beers_df_corrected['nbr_ratings'] = beers_df_corrected.index.map(beer_label_counts).fillna(0).astype(int)
    beers_df_corrected.reset_index(inplace=True)

    beers_df_corrected['location'] = beers_df_corrected['location'].astype(str)

    return beers_df_corrected
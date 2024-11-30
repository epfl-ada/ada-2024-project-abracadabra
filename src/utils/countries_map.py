import pandas as pd
import plotly.express as px
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os


#pip install geopandas matplotlib




def compute(breweries_df,beers_df, beers, labels, label_to_match):
    beers['Label'] = labels
    beers = beers.merge(beers_df[['id', 'brewery_id']], left_on='id_beer', right_on='id', how='left')

    amount_beers_per_country_labelled = compute_number_beers_per_country(breweries_df, beers, label_to_match)
    draw_map(amount_beers_per_country_labelled[['location','frequency']],'Beer Distribution by Country')

def compute_number_beers_per_country(breweries_df, beers, label_to_match):
    beers_labeled = beers[beers['Label'] == label_to_match]
    label_counts = beers_labeled.groupby('brewery_id').size().reset_index(name='amount_of_labels')
    breweries_df = breweries_df.merge(label_counts, how='left', left_on='id', right_on='brewery_id')
    breweries_df['amount_of_labels'] = breweries_df['amount_of_labels'].fillna(0).astype(int)
    breweries_df = breweries_df.drop(columns=['brewery_id'])
    breweries_df = match_countries(breweries_df)
    breweries_df = breweries_df.groupby('location').agg({'amount_of_labels': 'sum','nbr_beers': 'sum'}).reset_index()
    breweries_df['frequency'] = breweries_df.amount_of_labels/breweries_df.nbr_beers
    return breweries_df

def draw_map(countries_Beers_labelled,title):
    current_directory = os.getcwd()
    shapefile_path = current_directory+ "/src/utils/data/ne_110m_admin_0_countries.shp"

    world = gpd.read_file(shapefile_path)
    world = world.merge(countries_Beers_labelled, how='left', left_on='NAME', right_on='location')
    world['frequency'] = world['frequency'].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(column='frequency', cmap='OrRd', legend=True,legend_kwds={'label': "Number of Beers per Country"},missing_kwds={'color': 'lightgrey', 'label': 'No Data'},ax=ax)
    ax.set_title(title, fontsize=16)
    plt.show()

    unmatched_locations = countries_Beers_labelled[~countries_Beers_labelled['location'].isin(world['NAME'])]

    """
    print("Unmatched locations in countries_Beers_labelled:")
    print("In total there are ", len(countries_Beers_labelled), "locations who have a beer in the end ",len(unmatched_locations),"did not match")
    print(unmatched_locations)
    """

    unmatched_locations.to_csv('unmatched_locations.txt', index=False, sep='\t')
    world['NAME'].to_csv('Countries on Map.txt', index=False, sep='\t')

def match_countries(breweries_df):
    countries = ["United States", "Canada", "United Kingdom", "Australia", "Germany", "Italy"]

    breweries_df['location'] = breweries_df['location'].fillna('Unknown')

    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United States of America' if 'United States' in x else x)
    breweries_df['location'] = breweries_df['location'].apply(lambda x: 'United States of America' if 'Utah' in x else x)

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

    return breweries_df
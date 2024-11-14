import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations

### Used for Part 1:

def recompute_grade(df, min_grade_value = 1, max_grade_value = 5):
    '''
    Recomputes the grades for the different attributes.
    
    Parameters :
    - df: DataFrame containing user data.
    - min_grade_value: minimal grade wanted.
    - max_grade_value : maximal grade wanted.

    Returns :
    - DataFrame with grades on new scale. 
    '''
    attributes_of_interest = ['appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']
    dataset = ['rb','ad']

    min_max_values = df[df['dataset'].isin(dataset)].groupby('dataset')[attributes_of_interest].agg(['min', 'max'])

    for attribute in attributes_of_interest:
        for name in dataset:
            min_val = min_max_values.loc[name, (attribute, 'min')]
            max_val = min_max_values.loc[name, (attribute, 'max')]
            
            df.loc[df['dataset'] == name, attribute] = ((df.loc[df['dataset'] == name, attribute] -min_val)/(max_val-min_val)*(max_grade_value - min_grade_value) + min_grade_value)

    return df

# Part 3.1

def classify_user_rating_level(user_df, enthusiasts_level=20, connoisseur_level=100):
    '''
    Classify users into different rating levels based on the number of ratings they have provided in both webistes.
    
    Parameters :
    - user_df: DataFrame containing user data.
    - enthusiasts_level: Threshold for the number of ratings required to classify a user as 'enthusiasts' (default is 20).
    - connoisseur_level : Threshold for the number of ratings required to classify a user as 'connsoisseur' (default is 100).

    Returns :
    - DataFrame with an additional column 'rating_level' indicating the lever of rating of the user. 
    '''

    # Helper function to classify based on the number of ratings
    def __classify(nbr_ratings):
        if nbr_ratings <= enthusiasts_level:
            return 'novice'
        elif nbr_ratings <= connoisseur_level:
            return 'enthusiast'
        else:
            return 'connoisseur'
        
    # Apply classification to each user (each row of the DataFrame) based on their number of ratings
    user_df['rating_user_level'] = user_df['nbr_ratings'].apply(__classify)

    return user_df

def plot_category_distrib(df, category_column):
    """
        Plot the distribution of a categorical variable in a bar plot.
        
        Parameters:
        - data: DataFrame containing the categorical variable.
        - category_column: Name of the column containing the categorical variable.
    """

    # Set up a small figure plot
    plt.figure(figsize=(6, 4))

    # Use seaborn countplot to plot the distribution
    sns.countplot(data=df, x=category_column, palette='magma')

    # Add titles and labels
    plt.title(f'Distribution of {category_column}')
    plt.xlabel(category_column)
    plt.ylabel('Count')

    plt.show()

def compute_variance_per_attribute(ratings_df, attributes_of_interest):
    grouped_ratings = ratings_df.groupby('id_beer')
    return grouped_ratings[attributes_of_interest].var()


def PCA_plot(data):

    attributes_of_interest_PCA = ['var_appearance', 'var_aroma', 'var_palate', 'var_taste', 'var_overall', 'var_rating']

    data_for_pca = data[attributes_of_interest_PCA]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, edgecolors='k')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Beers (2D) advocate with rating')
    plt.grid(True)
    plt.show()

    principal_components_loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    print("principal components:",principal_components_loadings)
    print("Explained variance",explained_variance)

# Part 2
def filter_ratings(ratings_df, threshold, attributes):
    '''
    Filter the beer with too few ratings or textual reviews.

    Parameters :
    - ratings_df: DataFrame containing ratings and textual reviews data
    - threshold: Minimal number of specific attribute ratings required to select a beer for the analysis
    - attributes: Attributes for which we want to have a minimal number of rating to select a beer for the analysis

    Returns :
    - DataFrame filtered. Only ratings for which the beer has enough number of ratings are remaining. Furthermore, the returned DataFrame
    only contains the meaningful features/attributes and the beer id.
    '''

    # Keeping only the relevant column/features : attributes and id_beer
    columns_to_keep = attributes + ['id_beer']
    df_filtered = ratings_df[columns_to_keep]

    # Dropping the rows with nan values
    df_filtered = df_filtered.dropna(subset=attributes, how='any')
    print("Pourcentage of ratings remaining after dropping rows with nan values in selected attributes: {:.2f} %".format(100 * len(df_filtered)/len(ratings_df)))

    # Group the beer per id and compute the size of each group (number of filtered ratings for each beer)
    valid_ratings_count = df_filtered.groupby('id_beer').size()

    # Keep all ratings for which the beer has enough filtered ratings
    beer_remaining = valid_ratings_count[valid_ratings_count >= threshold]
    df_filtered = df_filtered[df_filtered['id_beer'].isin(beer_remaining.index)]
    print("Pourcentage of ratings remaining after dropping rating for which beer has too few valid ratings : {:.2f} %".format(100 * len(df_filtered)/len(ratings_df))) 

    return df_filtered

# Part 2
def classify_value_threshold(df, attributes_interest, attribute_labelling = ['overall'], threshold_controversial = 0.5, threshold_universal = 0.1):
    '''
    This function studies the variance in some attributes according to the label provided to the beers. Beers are labelled
    as controversial/universal according to the value of the variance of a certain attribute.

    Parameters :
    - df: DataFrame containing ratings data
    - attributes_interest: Attributes the function study the variance on
    - attribute_labelling: Attribute for which the variance define the label of the beer
    - threshold_controversial: Threshold for which a higher variance value labels the beer as controversial
    - threshold_universal: Threshold for which a lower variance value labels the beer as universal

    Returns :
    - The variance of the attributes of interest for both class of beers, controversial and universal 
    '''
    # Compute the variance of the attribute for labelling for each beer
    attribute_variance = compute_variance_per_attribute(df, attribute_labelling)

    # Extract beer id with controversial / universal overall variance
    controv_rating_id = attribute_variance[attribute_variance.values >= threshold_controversial]
    univ_rating_id = attribute_variance[attribute_variance.values < threshold_universal]

    # Print the distribution of class
    print("Percentage of beers with controversial overall variance : {:.2f} %".format(100 * len(controv_rating_id) / len(attribute_variance)))
    print("Percentage of beers with universal overall variance : {:.2f} %".format(100 * len(univ_rating_id) / len(attribute_variance)))

    # Filter the ratings for which the beers are defined as controversial or universal
    controv_rating = df[df['id_beer'].isin(controv_rating_id.index)]
    univ_rating = df[df['id_beer'].isin(univ_rating_id.index)]

    # Compute the variance across the attributes of interest on the different class for each beers
    controv_rating_variance_attribute = compute_variance_per_attribute(controv_rating, attributes_interest)
    univ_rating_variance_attribute = compute_variance_per_attribute(univ_rating, attributes_interest)

    return [controv_rating_variance_attribute, univ_rating_variance_attribute]

def classify_percentage_distribution(df, attributes_interest, attribute_labelling = ['overall'], threshold_percentage=10):
    '''
    This function does the same as classify_value_threshold(), but instead of keeping values below
    or above a certain threshold, it picks a certain low and high percentage of the distribution.
    This function studies the variance in some attributes according to the label provided to the beers. Beers are labelled
    as controversial/universal according to the distribution of their variance and are selected if in highest
    or lowest part of the distribution

    Parameters :
    - df: DataFrame containing ratings data
    - attributes_interest: Attributes the function study the variance on
    - attribute_labelling: Attribute for which the variance define the label of the beer
    - threshold_percentage: Value selecting the top and bottom x percentage of values in the distribution

    Returns :
    - The variance of the attributes of interest for both class of beers, controversial and universal 
    '''
    # Compute the variance of the attribute for labelling for each beer
    attribute_variance = compute_variance_per_attribute(df, attribute_labelling)

    # Computing the top and bottom value threshold for quantile x% and 1-x%
    top_threshold = attribute_variance.values.quantile(1-threshold_percentage/100)
    bottom_threshold = attribute_variance.values.quantile(threshold_percentage/100)

    # Extract beer id with controversial / universal overall variance
    controv_rating_id = attribute_variance[attribute_variance.values >= top_threshold]
    univ_rating_id = attribute_variance[attribute_variance.values <= bottom_threshold]

    # Asserting the distribution of the classes
    print("Percentage of beers with controversial overall variance : {:.2f} %".format(100 * len(controv_rating_id) / len(attribute_variance)))
    print("Percentage of beers with universal overall variance : {:.2f} %".format(100 * len(univ_rating_id) / len(attribute_variance)))

    # Filter the ratings for which the beers are defined as controversial or universal
    controv_rating = df[df['id_beer'].isin(controv_rating_id.index)]
    univ_rating = df[df['id_beer'].isin(univ_rating_id.index)]

    # Compute the variance across the attributes of interest on the different class for each beers
    controv_rating_variance_attribute = compute_variance_per_attribute(controv_rating, attributes_interest)
    univ_rating_variance_attribute = compute_variance_per_attribute(univ_rating, attributes_interest)

    return [controv_rating_variance_attribute, univ_rating_variance_attribute]

def compute_similarity_scores(df):
    sentiment_columns = ['class_sentiment_bert', 'class_sentiment_google', 'class_sentiment_distilbert',
                         'class_sentiment_GPT']

    for col1, col2 in combinations(sentiment_columns, 2):
        df[f'exact_similarity_{col1}_{col2}'] = (df[col1] == df[col2]).astype(int)

        df[f'plus_minus_1_similarity_{col1}_{col2}'] = (abs(df[col1] - df[col2]) <= 1).astype(int)

    exact_similarity_columns = [f'exact_similarity_{col1}_{col2}' for col1, col2 in combinations(sentiment_columns, 2)]
    plus_minus_1_similarity_columns = [f'plus_minus_1_similarity_{col1}_{col2}' for col1, col2 in
                                       combinations(sentiment_columns, 2)]

    df['exact_similarity_score'] = df[exact_similarity_columns].sum(axis=1)
    df['plus_minus_1_similarity_score'] = df[plus_minus_1_similarity_columns].sum(axis=1)

    return df, exact_similarity_columns, plus_minus_1_similarity_columns

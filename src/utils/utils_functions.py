import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from itertools import combinations


### Used for Part 1:

def recompute_grade(df, min_grade_value=1, max_grade_value=5, attributes_of_interest = ['appearance', 'aroma', 'palate', 'taste', 'overall']):
    '''
    Recomputes the grades for the different attributes.
    
    Parameters :
    - df: DataFrame containing user data.
    - min_grade_value: minimal grade wanted.
    - max_grade_value : maximal grade wanted.
    - attributes_of_interest: attributes over which which should recompute the grades

    Returns :
    - DataFrame with grades on new scale. 
    '''

    dataset = ['rb', 'ad']

    min_max_values = df[df['dataset'].isin(dataset)].groupby('dataset')[attributes_of_interest].agg(['min', 'max'])

    for attribute in attributes_of_interest:
        for name in dataset:
            min_val = min_max_values.loc[name, (attribute, 'min')]
            max_val = min_max_values.loc[name, (attribute, 'max')]

            df.loc[df['dataset'] == name, attribute] = (
                    (df.loc[df['dataset'] == name, attribute] - min_val) / (max_val - min_val) * (
                    max_grade_value - min_grade_value) + min_grade_value)

    return df


def recompute_grade_with_sent(df, min_grade_value=1, max_grade_value=5):
    '''
    Recomputes the grades for the different attributes.
    
    Parameters :
    - df: DataFrame containing user data.
    - min_grade_value: minimal grade wanted.
    - max_grade_value : maximal grade wanted.

    Returns :
    - DataFrame with grades on new scale. 
    '''
    attributes_of_interest = ['appearance', 'aroma', 'palate', 'taste', 'overall', 'sentiment_bert']
    dataset = ['rb', 'ad']

    min_max_values = df[df['dataset'].isin(dataset)].groupby('dataset')[attributes_of_interest].agg(['min', 'max'])

    for attribute in attributes_of_interest:
        for name in dataset:
            min_val = min_max_values.loc[name, (attribute, 'min')]
            max_val = min_max_values.loc[name, (attribute, 'max')]

            df.loc[df['dataset'] == name, attribute] = (
                    (df.loc[df['dataset'] == name, attribute] - min_val) / (max_val - min_val) * (
                    max_grade_value - min_grade_value) + min_grade_value)

    return df


def PCA_plot(data, attributes_of_interest_PCA=['appearance', 'aroma', 'palate', 'taste', 'overall']):
    '''
    Plots the PCA on the variance of the attributes of the beers, we reduce the dimensions on 2D.
    Also prints the aigenvalues and eigenvector.

    Parameters :
    - df: DataFrame containing the variance data
    - attributes_of_interest_PCA: Attributes we chose to analyse the variance from
    '''
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

    plt.figure(figsize=(10, 6))
    plt.hexbin(pca_df['PC1'], pca_df['PC2'], gridsize=30, cmap='Blues', edgecolors='k')
    plt.colorbar(label='Counts')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Beers (Density Plot)')
    plt.grid(True)
    plt.show()

    principal_components_loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    print("principal components:", principal_components_loadings)
    print("Explained variance", explained_variance)


# Part 2
def filter_ratings_new(ratings_df, beers_df, breweries_df, users_df, threshold, attributes, verbose=True):
    '''
    Filter the beer with too few ratings or textual reviews.

    Parameters :
    - ratings_df: DataFrame containing ratings and textual reviews data
    - beers_df: DataFrame containing beers data
    - breweries_df: DataFrame containing breweries data
    - threshold: Minimal number of specific attribute ratings required to select a beer for the analysis
    - attributes: Attributes for which we want to have a minimal number of rating to select a beer for the analysis
    - verbose: Wheter to print the percentage of ratings and beer remaining after filtering


    Returns :
    - df_filtered : DataFrame filtered. Only ratings for which the beer has enough number of ratings are remaining. Furthermore, the returned DataFrame
    only contains the meaningful features/attributes and the beer id.
    - df_breweries_filtered: DataFrame containing only the breweries remaining after filetering
    - beers_df: DataFrame containing only the beers remaining after filetering
    '''

    #Deep copies just to ensure we do not change the datasets of origin
    ratings_df = ratings_df.copy(deep = True)
    beers_df = beers_df.copy(deep=True)
    breweries_df = breweries_df.copy(deep = True)
    users_df = users_df.copy(deep = True)

    #Computes length of origin for future computation
    init_length_ratings = len(ratings_df)
    init_length_beers = len(beers_df)
    init_length_brew = len(breweries_df)
    init_length_user = len(users_df)

    # Keeping only the relevant column/features : attributes and id_beer
    columns_to_keep = attributes + ['id_beer'] + ['id_user'] + ['id_brewery']
    df_filtered = ratings_df[columns_to_keep]

    # Dropping the rows with nan values
    df_filtered = df_filtered.dropna(subset=attributes, how='any')
    beer_remaining = df_filtered.groupby('id_beer').size()
    df_filtered = df_filtered[df_filtered['id_beer'].isin(beer_remaining.index)]
    
    #Filters the remaining beers, breweries and users. Recomputes the true number of ratings just to ensure they are correct
    beers_df = beers_df[beers_df['id'].isin(beer_remaining.index)]
    beers_df.loc[:, 'true_number_ratings'] = beers_df['id'].map(beer_remaining)

    valid_beers_count = beers_df.groupby('brewery_id').size()
    breweries_df = breweries_df[breweries_df.id.isin(valid_beers_count.index)]
    breweries_df.loc[:, 'true_number_beers'] = breweries_df['id'].map(valid_beers_count)

    valid_users_count = df_filtered.groupby('id_user').size()
    users_df = users_df[users_df.id.isin(valid_users_count.index)]
    users_df.loc[:, 'true_number_ratings'] = users_df['id'].map(valid_users_count)

    if verbose :
        print(
        "Percentage of ratings remaining after dropping rows with nan values in selected attributes: {:.2f} %".format(
            100 * len(df_filtered) / init_length_ratings))
        print(
        "Percentage of beers remaining after dropping ratings with nan values in selected attributes: {:.2f} %".format(
            100 * len(beers_df) / init_length_beers))
        print(
        "Percentage of breweries remaining after dropping ratings with nan values in selected attributes: {:.2f} %".format(
            100 * len(breweries_df) / init_length_brew))
        print(
        "Percentage of users remaining after dropping ratings with nan values in selected attributes: {:.2f} %".format(
            100 * len(users_df) / init_length_user))
    
    #Computes the length for future purposes
    init_length_ratings_no_nan = len(df_filtered)
    init_length_beers_no_nan = len(beer_remaining)
    init_length_brew_no_nan = len(breweries_df)
    init_length_user_no_nan = len(users_df)

    # KEEPING BEER FOR WHICH NB RATING ABOVE CERTAIN THRESHOLD
    beer_remaining = beer_remaining[beer_remaining >= threshold]
    df_filtered = df_filtered[df_filtered['id_beer'].isin(beer_remaining.index)]

    beers_df = beers_df[beers_df['id'].isin(beer_remaining.index)]
    beers_df.loc[:, 'true_number_ratings'] = beers_df['id'].map(beer_remaining)
    
    valid_beers_count = beers_df.groupby('brewery_id').size()
    breweries_df = breweries_df[breweries_df.id.isin(valid_beers_count.index)]
    breweries_df.loc[:, 'true_number_beers'] = breweries_df['id'].map(valid_beers_count)

    valid_users_count = df_filtered.groupby('id_user').size()
    users_df = users_df[users_df.id.isin(valid_users_count.index)]
    users_df.loc[:, 'true_number_ratings'] = users_df['id'].map(valid_users_count)

    if verbose :
        print(
        "\nPercentage of ratings remaining after dropping beers with too few ratings from the nan-filtered dataset: {:.2f} %".format(
            100 * len(df_filtered) / init_length_ratings_no_nan))
        print(
        "Percentage of beers remaining after dropping beers with too few ratings from the nan-filtered dataset: {:.2f} %".format(
            100 * len(beers_df) / init_length_beers_no_nan))
        print(
        "Percentage of breweries remaining after dropping beers with too few ratings from the nan-filtered dataset: {:.2f} %".format(
            100 * len(breweries_df) / init_length_brew_no_nan))
        print(
        "Percentage of users remaining after dropping beers with too few ratings from the nan-filtered dataset: {:.2f} %".format(
            100 * len(users_df) / init_length_user_no_nan))
    
    # Rename and set as the new real number of ratings
    beers_df = beers_df.drop(columns='nbr_ratings').rename(columns={'true_number_ratings':'nbr_ratings'})
    breweries_df = breweries_df.drop(columns='nbr_beers').rename(columns={'true_number_beers':'nbr_beers'})
    users_df = users_df.drop(columns='nbr_ratings_total').rename(columns={'true_number_ratings':'nbr_ratings_total'})
    
    return df_filtered, breweries_df, beers_df, users_df


def filter_ratings(ratings_df, threshold, attributes, verbose=True):
    '''
    Filter the beer with too few ratings or textual reviews.

    Parameters :
    - ratings_df: DataFrame containing ratings and textual reviews data
    - threshold: Minimal number of specific attribute ratings required to select a beer for the analysis
    - attributes: Attributes for which we want to have a minimal number of rating to select a beer for the analysis
    - verbose: Wheter to print the percentage of ratings and beer remaining after filtering

    Returns :
    - DataFrame filtered. Only ratings for which the beer has enough number of ratings are remaining. Furthermore, the returned DataFrame
    only contains the meaningful features/attributes and the beer id.
    '''

    # Keeping only the relevant column/features : attributes and id_beer
    columns_to_keep = attributes + ['id_beer']
    df_filtered = ratings_df[columns_to_keep]


    # Dropping the rows with nan values
    df_filtered = df_filtered.dropna(subset=attributes, how='any')

    nb_original_ratings_no_nan = len(df_filtered)
    nb_original_beer_no_nan = len(df_filtered['id_beer'].unique())

    # Print percentage of beer dropped due to nan values in the ratings
    # print(
    #    "Percentage of ratings remaining after dropping rows with nan values in selected attributes: {:.2f} %".format(
    #        100 * len(df_filtered) / len(ratings_df)))

    # Group the beer per id and compute the size of each group (number of filtered ratings for each beer)
    valid_ratings_count = df_filtered.groupby('id_beer').size()

    # Keep all ratings for which the beer has enough filtered ratings
    beer_remaining = valid_ratings_count[valid_ratings_count >= threshold]
    df_filtered = df_filtered[df_filtered['id_beer'].isin(beer_remaining.index)]
    modified_nb_ratings = len(df_filtered)
    modified_nb_beer = len(beer_remaining)
    if verbose:
        print(
        "Percentage of ratings remaining : {:.2f} % and beers remaining : {:.2f} % after dropping rows with nan values and rating for which beer has less than {} valid ratings".format(
            100 * modified_nb_ratings / nb_original_ratings_no_nan,
            100 * modified_nb_beer/nb_original_beer_no_nan,
            threshold))

    return df_filtered


# May be useful later
# def drop_nan_ratings(ratings_df):
#     df_filtered = ratings_df.dropna(how='any')
#     print("Percentage of ratings remaining after dropping nan values {}%".format(len(df_filtered)/len(ratings_df)))
#     return df_filtered


def plot_threshold_filtering(ratings_df, thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200], verbose = False):
    '''
    Plot the remaining percentager of ratings and beers as a function of threshold filtering value
    Percentage is compared to ratings and beers without nan values and at least one rating

    Parameters :
    - ratings_df: DataFrame containing ratings and textual reviews data
    - thresholds: List of threshold representing minimal number of specific attribute ratings required to select a beer for the analysis
    - verbose: Wheter to print the remaining percentage of ratings after each filtering

    Returns :
    - None
    '''
    # Filtering nan and ensuring beers with at least one rating for percentage comparison purpose 
    ratings_df = filter_ratings(ratings_df, 1, ['appearance', 'aroma', 'palate', 'taste','overall'], verbose=False)

    values = []

    # Original number of ratings and beers for comparison
    original_nb_ratings = len(ratings_df)
    original_nb_beers = len(ratings_df['id_beer'].unique())

    # Applying the filter and storing the percentage of ratings and beers remaining for each threshold
    for t in thresholds:
        filtered_ratings = filter_ratings(ratings_df, t, ['appearance', 'aroma', 'palate', 'taste','overall'], verbose=verbose)
        values.append({
            'threshold': t, 
            'percentage_rating': 100*len(filtered_ratings)/original_nb_ratings, 
            'percentage_beer': 100*len(filtered_ratings['id_beer'].unique())/original_nb_beers})
        #print("Percentage of beer remaining {} %".format(100*len(filtered_ratings['id_beer'].unique())/original_nb_beers))

    values_df = pd.DataFrame(values)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Threshold')
    ax1.set_title('Effect of threshold on ratings and beers filtering')
    ax1.grid(True)

    # Plot remaining ratings
    ax1.plot(values_df['threshold'], values_df['percentage_rating'], 'b-o', label='Remaining Ratings (%)')
    ax1.set_ylabel('Percentage of Remaining Ratings', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper right')

    # Second y-axis for remaining beers
    ax2 = ax1.twinx()  # Second y-axis sharing the same x-axis
    ax2.plot(values_df['threshold'], values_df['percentage_beer'], 'r-s', label='Remaining Beers (%)')
    ax2.set_ylabel('Percentage of Remaining Beers', color='r')
    ax2.tick_params(axis='y', labelcolor='r')    
    ax2.legend(loc='upper center')

    plt.tight_layout()
    plt.show()

    


# Part 2
def classify_value_threshold(df, attributes_interest, attribute_labelling=['overall'], threshold_controversial=1,
                             threshold_universal=0.1):
    '''
    This function studies the variance in some attributes according to the label provided to the beers. Beers are labelled
    as controversial/universal according to the value of the variance of a certain attribute.

    Parameters :
    - df: DataFrame containing ratings data
    - attributes_interest: Attributes the function computes the variance on
    - attribute_labelling: Attribute for which the variance defines the label of the beer
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
    print("Percentage of beers with controversial overall variance : {:.2f} %".format(
        100 * len(controv_rating_id) / len(attribute_variance)))
    print("Percentage of beers with universal overall variance : {:.2f} %".format(
        100 * len(univ_rating_id) / len(attribute_variance)))

    # Filter the ratings for which the beers are defined as controversial or universal
    controv_rating = df[df['id_beer'].isin(controv_rating_id.index)]
    univ_rating = df[df['id_beer'].isin(univ_rating_id.index)]

    # Compute the variance across the attributes of interest on the different class for each beers
    controv_rating_variance_attribute = compute_variance_per_attribute(controv_rating, attributes_interest)
    univ_rating_variance_attribute = compute_variance_per_attribute(univ_rating, attributes_interest)

    return [controv_rating_variance_attribute, univ_rating_variance_attribute]


def classify_percentage_distribution(df, attributes_interest, attribute_labelling=['overall'], threshold_percentage=10):
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
    # Need attribute_labelling[0] as the parameter is entered in the syntax ['attribute_name'] and attribute_variance output a DataFrame
    # --> attribute_variance[attribute_labelling[0]] corresponds to attribute_variance['attribute_name']
    # This could be done differently but this is chosen for consistent syntax in the different functions
    top_threshold = attribute_variance[attribute_labelling[0]].quantile(1-threshold_percentage/100)
    bottom_threshold = attribute_variance[attribute_labelling[0]].quantile(threshold_percentage/100)

    # Extract beer id with controversial / universal overall variance
    controv_rating_id = attribute_variance[attribute_variance[attribute_labelling[0]] >= top_threshold]
    univ_rating_id = attribute_variance[attribute_variance[attribute_labelling[0]] <= bottom_threshold]

    # Asserting the distribution of the classes
    print("Percentage of beers with controversial overall variance : {:.2f} %".format(
        100 * len(controv_rating_id) / len(attribute_variance)))
    print("Percentage of beers with universal overall variance : {:.2f} %".format(
        100 * len(univ_rating_id) / len(attribute_variance)))

    # Filter the ratings for which the beers are defined as controversial or universal
    controv_rating = df[df['id_beer'].isin(controv_rating_id.index)]
    univ_rating = df[df['id_beer'].isin(univ_rating_id.index)]

    # Compute the variance across the attributes of interest on the different class for each beers
    controv_rating_variance_attribute = compute_variance_per_attribute(controv_rating, attributes_interest)
    univ_rating_variance_attribute = compute_variance_per_attribute(univ_rating, attributes_interest)

    return [controv_rating_variance_attribute, univ_rating_variance_attribute]


def t_test_statistic(df, attributes_of_interest=['appearance', 'aroma', 'palate', 'taste', 'overall']):
    '''
    This function performs a t test. We want to test the mean of the variance of the different attributes.
    The H0 hypothesis is that the true mean of the variance of a given attribute between the different bears are equal.
    H1 is that the means are different. Furthermore we plot in a heatmap the different values obtained between the variance of the attributes.


    Parameters :
    - df: DataFrame containing the variance data
    - attributes_interest: Attributes we chose to analyse the variance from
    '''

    #Initialize bunch of variables used to store the results
    p_value_table = np.zeros((len(attributes_of_interest), len(attributes_of_interest)))
    ci_table = np.zeros((len(attributes_of_interest), len(attributes_of_interest), 2))
    mean_value_mean = np.zeros((len(attributes_of_interest), len(attributes_of_interest)))
    annotations = np.empty(p_value_table.shape, dtype=object)

    #Perform a double for loop to do the t test between the distribution of the various attributes
    for i, attribute1 in enumerate(attributes_of_interest):
        for j, attribute2 in enumerate(attributes_of_interest):
            attribute1_variance = df[attribute1]
            attribute2_variance = df[attribute2]

            #Perform T test over the two attributes of the forloop
            ttest_result = ttest_ind(attribute1_variance, attribute2_variance)
            ci_t_test_result = ttest_result.confidence_interval(confidence_level=0.95)
            p_value_table[i, j] = ttest_result.pvalue
            ci_table[i, j, 0] = ci_t_test_result.low
            ci_table[i, j, 1] = ci_t_test_result.high
            mean_value_mean[i, j] = attribute2_variance.mean()
            annotations[i, j] = f"({mean_value_mean[i, j]:.2f} ± {ci_table[i, j, 0]:.2f}, {ci_table[i, j, 1]:.2f})"

    p_value_df = pd.DataFrame(p_value_table, index=attributes_of_interest, columns=attributes_of_interest)

    #Plot results
    plt.figure(figsize=(8, 6))
    sns.heatmap(p_value_df, annot=True, cmap="Reds", vmin=0, vmax=1, square=True, cbar_kws={'label': 'P-Value'})
    plt.title("P-Value heatmap for the T Test on the variance of the different attributes")
    plt.show()

def single_t_test_statistic_(df, attributes_of_interest=['appearance', 'aroma', 'palate', 'taste', 'overall'], attribute_single = 'sentiment_bert'):
    '''
    This function performs a t test. We want to test the mean of the variance of the different attributes.
    The H0 hypothesis is that the true mean of the variance of a given attribute between the different bears are equal.
    H1 is that the means are different. Furthermore we plot in a heatmap the different values obtained between the variance of the attributes.


    Parameters :
    - df: DataFrame containing the variance data
    - attributes_interest: Attributes we chose to analyse the variance from
    '''
    p_value_table = np.zeros((len(attributes_of_interest), 1))
    ci_table = np.zeros((len(attributes_of_interest), 1, 2))
    mean_value_mean = np.zeros((len(attributes_of_interest), 1))
    annotations = np.empty(p_value_table.shape, dtype=object)

    for i, attribute1 in enumerate(attributes_of_interest):
        ttest_result = ttest_ind(df[attribute1], df[attribute_single])

        ci_t_test_result = ttest_result.confidence_interval(confidence_level=0.95)
        p_value_table[i, 0] = ttest_result.pvalue
        ci_table[i, 0, 0] = ci_t_test_result.low
        ci_table[i, 0, 1] = ci_t_test_result.high
        mean_value_mean[i, 0] = df[attribute_single].mean()
        annotations[i, 0] = f"({mean_value_mean[i, 0]:.2f} ± {ci_table[i, 0, 0]:.2f}, {ci_table[i, 0, 1]:.2f})"
    

    p_value_df = pd.DataFrame(p_value_table, index=attributes_of_interest, columns=[attribute_single])

    p_value_df.plot(kind="bar", legend=False)
    plt.title("P-Values by Attribute")
    plt.ylabel("P-Value")
    plt.xlabel("Attributes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compute_similarity_scores(df):
    """
        Calculate similarity scores between sentiment classification columns in a DataFrame.

        This function compares pairs of sentiment classifications, both for exact matches and for approximate matches
        (allowing a difference of ±1 between classes). It adds columns indicating pairwise similarity and
        computes aggregate similarity scores for each row.

        Parameters:
        - df: pandas DataFrame containing reviews and sentiment classification columns.

        Returns:
        - df: Updated DataFrame containing the new similarity columns and aggregate similarity scores.
        - exact_similarity_columns: List of column names for exact match similarity indicators.
        - plus_minus_1_similarity_columns: List of column names for ±1 match similarity indicators.
        """
    sentiment_columns = ['class_sentiment_bert', 'class_sentiment_google', 'class_sentiment_distilbert',
                         'class_sentiment_GPT']

    exact_similarity_columns = []
    plus_minus_1_similarity_columns = []

    for col1, col2 in combinations(sentiment_columns, 2):
        exact_col_name = f'exact_similarity_{col1.split("_")[-1]}_{col2.split("_")[-1]}'
        plus_minus_col_name = f'plus_minus_1_similarity_{col1.split("_")[-1]}_{col2.split("_")[-1]}'

        df[exact_col_name] = (df[col1] == df[col2]).astype(int)
        df[plus_minus_col_name] = (abs(df[col1] - df[col2]) <= 1).astype(int)

        exact_similarity_columns.append(exact_col_name)
        plus_minus_1_similarity_columns.append(plus_minus_col_name)

    df['exact_similarity_score'] = df[exact_similarity_columns].sum(axis=1)
    df['plus_minus_1_similarity_score'] = df[plus_minus_1_similarity_columns].sum(axis=1)

    return df, exact_similarity_columns, plus_minus_1_similarity_columns


def plot_sentiment_similarities(df, exact_similarity_columns, plus_minus_1_similarity_columns):
    """
        Plots the similarity scores between pairs of sentiment analysis models.

        Parameters:
            -df : DataFrame containing similarity data between model pairs
            -exact_similarity_columns : A list of the column names in `df` representing exact similarity scores between pairs of models.
            -plus_minus_1_similarity_columns : A list of the column names in `df` representing ±1 similarity scores between pairs of models.
    """

    plt.figure(figsize=(10, 6))
    ax1 = df[exact_similarity_columns].sum().plot(kind='bar', color='skyblue')
    plt.title('Exact similarity scores between pairs of models')
    plt.ylabel('Number of matches')
    plt.xlabel('Model pair')
    plt.xticks(rotation=45)

    for i, value in enumerate(df[exact_similarity_columns].sum()):
        ax1.text(i, value + 0.5, str(round(value / 6000, 2)), ha='center', va='bottom')

    plt.show()

    plt.figure(figsize=(10, 6))
    ax2 = df[plus_minus_1_similarity_columns].sum().plot(kind='bar', color='lightcoral')
    plt.title('±1 Similarity scores between pairs of models')
    plt.ylabel('Number of matches')
    plt.xlabel('Model pair')
    plt.xticks(rotation=45)

    for i, value in enumerate(df[plus_minus_1_similarity_columns].sum()):
        ax2.text(i, value + 0.5, str(round(value / 6000, 2)), ha='center', va='bottom')

    plt.show()


def max_min_variance_count(df):
    """
    Find the attributes with maximal and minimal variance attribute across all beers

    Parameters:
    - df : DataFrame containing the variance of the attributes

    Returns :
    - Two Series containing the name of the attributes with maximal and minimal variance for each beer
    """
    # Get the name of the column having the maximal/minimal variance for each row
    max_var_attribute = df.idxmax(axis = 1)
    min_var_attribute = df.idxmin(axis = 1)

    # Print the distribution of attributes having max and min variance per beer
    print("Maximal variance attribute count :", max_var_attribute.value_counts())
    print("Minimal variance attriubte count : ", min_var_attribute.value_counts())

    return [max_var_attribute, min_var_attribute]

def plot_count_max_min_variance_count(max_var_count, min_var_count, classification='all'):
    plt.subplot(2, 1, 1)
    plt.hist(max_var_count)
    plt.title(f"Distribution of maximal variance attribute across {classification} beers")
    plt.xlabel("Attributes")
    plt.ylabel("Max variance count")

    plt.subplot(2, 1, 2)
    plt.hist(min_var_count)
    plt.title(f"Distribution of minimal variance attribute across {classification} beers")
    plt.xlabel("Attributes")
    plt.ylabel("Min variance count")

    plt.tight_layout()
    plt.show()

def compute_correlation(df_variances, attribute_corr):
    correlations = df_variances.corr()[attribute_corr[0]]
    return correlations.drop(attribute_corr)

def plot_correlation(correlations, title= 'Correlation of attributes variances with Overall variance'):
    # Plot the correlation
    plt.figure(figsize=(8, 6))
    correlations.plot(kind='bar', color=['blue' if c > 0 else 'red' for c in correlations])
    plt.title(title)
    plt.xlabel('Attributes')
    plt.ylabel('Correlation')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()

def compute_variance_per_attribute(ratings_df, attributes_of_interest):
    '''
    Computes the variance of desired attributes for each beer

    Parameters :
    - ratings_df: DataFrame containing the ratings
    - attributes_of_interest: Attributes the function computes the variance on

    Returns :
    - A DataFrame with the variance of the attributes of interest for each beer
    '''
    grouped_ratings = ratings_df.groupby('id_beer')
    return grouped_ratings[attributes_of_interest].var()

def compute_stastics(ratings_df, attributes = ['aroma','appearance', 'palate','taste','overall'], source = ['ad','rb']):
    '''
    Computes the statistics of the different attributes distribution

    Parameters :
    - ratings_df: DataFrame containing the ratings
    - attributes: Attributes the function computes the statistics on
    - source: The dataset to compute the statistic for  
    '''
    stats = {}
    for dataset in source:
        stats[dataset] = {}
        new_ratings = ratings_df[ratings_df['dataset']==dataset]
        for attrib in attributes:
            data = new_ratings[attrib].dropna()
            resolution = np.diff(np.sort(data.unique())).min() #Computes the resolution of a grade
            stats[dataset][attrib] = {'min': data.min(), 'max': data.max(), 'median': data.median(), 'resolution': resolution}

    print(stats)

def plot_var_distrib_violin_grades(ratings_df):
    ratings_advocate = ratings_df[ratings_df['dataset']=='ad']
    ratings_ratebeer = ratings_df[ratings_df['dataset']=='rb']

    ratings_advocate = ratings_advocate[['aroma','appearance', 'palate','taste','overall','appearance']]
    ratings_ratebeer = ratings_ratebeer[['aroma','appearance', 'palate','taste','overall','appearance']]

    sns.violinplot(ratings_ratebeer)
    plt.title("Distribution of grades across various attributes on the ratebeer dataset")
    plt.xlabel("Attributes")
    plt.ylabel("Grade")
    plt.show()

    sns.violinplot(ratings_advocate)
    plt.title("Distribution of grades across various attributes on the advocate dataset")
    plt.xlabel("Attributes")
    plt.ylabel("Grade")
    plt.show()

    del ratings_advocate
    del ratings_ratebeer

def plot_var_distrib_violin(var_df):
    sns.violinplot(var_df)
    plt.title("Comparison of variance distribution across attributes")
    plt.xlabel("Attributes")
    plt.ylabel("Variance")
    plt.show()

def plot_overall_hist_distrib(var_df):
    plt.hist(var_df, bins = 40)
    plt.title("Distribution of overall variance across all beers")
    plt.xlabel("Variance")
    plt.ylabel("Count")
    plt.show()

def plot_var_boxplot(controv_df, univ_df):
    plt.subplot(2, 1, 1)
    sns.boxplot(controv_df)
    plt.title("Distribution of variance across attributes for beers with high variance of overall score (labelled controversial)")
    plt.xlabel("Attributes")
    plt.ylabel("Variance distribution")
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])

    plt.subplot(2, 1, 2)
    sns.boxplot(univ_df)
    plt.title("Distribution of variance across attributes for beers with low variance of overall score (labelled universal)")
    plt.xlabel("Attributes")
    plt.ylabel("Variance distribution")
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    plt.tight_layout()
    plt.show()

def plot_histogram_nbr_ratings_total(users_df):
    '''
    Makes a plot of the distribution of the number of users per number of ratings made

    Parameters :
    - users_df: Dataframe with users data

    '''
    #Uses the code of the Exercise 2: Becoming a DataVizard solution.ipynb
    array_1000 = plt.hist(users_df.nbr_ratings_total,bins=10000,log=True,histtype='step')
    plt.close()

    plt.loglog(array_1000[1][1:],array_1000[0])
    plt.title('Distribution of total number of ratings for a user', fontsize=16)
    plt.xlabel('Number of Ratings in logscale', fontsize=14)
    plt.ylabel('Number of Users in logscale', fontsize=14)

    plt.show()

    print(users_df.nbr_ratings_total.describe())


def plot_threshold_variance_variation(ratings_df):
    # TODO LATER WITH FILTER_RATINGS NEW INSTEAD
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200]

    n_rows = 3
    n_cols = math.ceil(len(thresholds) / n_rows)
    
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10), sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, threshold in enumerate(thresholds):
        filtered_ratings = filter_ratings(ratings_df, threshold, ['appearance', 'aroma', 'palate', 'taste','overall'])
        var_attr = compute_variance_per_attribute(filtered_ratings, ['appearance', 'aroma', 'palate', 'taste','overall'])

        sns.violinplot(data=var_attr, ax=axs[i])
        axs[i].set_title(f"Threshold {threshold}")
        axs[i].set_xlabel("Attributes")

    plt.tight_layout()
    plt.show()

def plot_threshold_variance_variation_with_sent(ratings_df):
    # TODO LATER WITH FILTER_RATINGS NEW INSTEAD
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 200]

    n_rows = 3
    n_cols = math.ceil(len(thresholds) / n_rows)
    
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10), sharey=True)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, threshold in enumerate(thresholds):
        filtered_ratings = filter_ratings(ratings_df, threshold, ['appearance', 'aroma', 'palate', 'taste','overall', 'sentiment_bert'])
        var_attr = compute_variance_per_attribute(filtered_ratings, ['appearance', 'aroma', 'palate', 'taste','overall', 'sentiment_bert'])

        sns.violinplot(data=var_attr, ax=axs[i])
        axs[i].set_title(f"Threshold {threshold}")
        axs[i].set_xlabel("Attributes")

    plt.tight_layout()
    plt.show()

def plot_variance_evolution_batches_barplot(ratings_df, beers_df, attributes=('overall')):
    '''
    Plots the evolution of the standard deviation of the variance of the beer ratings which are batched together depending on the number of ratings for a beer. 

    Parameters :
    - ratings_df: DataFrame containing the ratings
    - beers_df: DataFrame containing the beers
    - attributes: the attribute over which we compute the variance of the grade. Default: overall
    '''
    #Deep copy for security
    ratings_df = ratings_df[[attributes, 'id', 'id_beer']].copy(deep=True)
    beers_df = beers_df.copy(deep=True)

    #Compute the variance of the beers for the said attribute. Not necessary to do over id to get size but works fine.
    ratings_df = ratings_df.groupby('id_beer').agg({attributes: 'var', 'id': 'size'}).rename(columns={'id': 'number_ratings'})

    #Give batches for each beer
    batch_edges = np.arange(0, 205, 5)
    ratings_df['batch'] = pd.cut(ratings_df['number_ratings'], bins=batch_edges, right=False, labels=batch_edges[:-1])
    ratings_df = ratings_df.rename(columns={'batch': 'batched_number_of_ratings'})

    #Computes std and mean of the variance of the ratings for the plot. In the end we do not use the mean. 
    batch_means = ratings_df.groupby('batched_number_of_ratings').agg({attributes: ['mean', 'std']})
    batch_means.columns = ['mean_' + attributes, 'std_' + attributes] #Rename

    batch_means = batch_means.reset_index()
    batch_means['batched_number_of_ratings'] = batch_means['batched_number_of_ratings'].astype(float) + 2.5

    #Bar plot
    plt.figure(figsize=(12, 6))
    x_positions = batch_means['batched_number_of_ratings']
    plt.bar(
        x_positions,#Use x_positions for clarity
        batch_means['std_' + attributes],
        width=4,#Tried 4, 5 everything is glued together.
        color='green',
        alpha=0.7,
        label=f'Standard Deviation {attributes.capitalize()}',
    )
    plt.xticks(x_positions, labels=x_positions, rotation=45)
    plt.title(f'Variance evolution with the number of reviews on the {attributes.capitalize()} attribute')
    plt.xlabel('Number of reviews')
    plt.ylabel(f'{attributes.capitalize()}')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()

    del ratings_df, beers_df,batch_means

def plot_attribute_variance_distrib_per_label(variances_df, labels, label_list=[1, 2, 0]):
    '''
    Plots the distribution of the variances of the main attributes for each class. 

    Parameters :
    - variances_df: DataFrame containing the variances of each attributes for each beer
    - labels: class of each beer
    - label_list: list to tell us which label corresponds to which class for the 1st GMM The list should give the label which corresponds to the following:[universal, neutral, controversial]
    '''

    variances_df = variances_df.copy(deep=True)
    variances_df['labels'] = labels  # Apply labels
    class_name = ['Controversial', 'Universal', 'Neutral']

    fig, axes = plt.subplots(nrows=1, ncols=len(label_list), figsize=(16, 6), sharey=True)
    for i, label in enumerate(label_list):
        # Group the beer (and variances) per class to plot their distribution
        variances_grouped = variances_df[variances_df.labels == label]

        # Create a boxplot for each attribute
        sns.boxplot(variances_grouped[['appearance', 'aroma', 'palate', 'taste', 'overall']], ax=axes[i])
        axes[i].set_title(f"{class_name[label]} class")
        axes[i].set_xlabel("Attributes")
        axes[i].set_ylabel("Variance" if i == 0 else '')  # Show ylabel only for the first subplot
        axes[i].grid(visible=True, which='major', linestyle='--')  # Show the grid
    
    plt.suptitle("Distribution of variances of the attributes for each class")
    plt.tight_layout()
    plt.show()

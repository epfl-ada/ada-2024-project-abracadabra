import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def compute_variance_per_attribute(data):
    attributes_of_interest = ['appearance', 'aroma', 'palate', 'taste', 'overall', 'rating']

    for attribute in attributes_of_interest:

        attribute_data = data.groupby("id_beer")[attribute].std()
        attribute_data.name = 'var_'+attribute
        data = data.merge(attribute_data, on='id_beer', how='left') #the sem needs at least 2 reviews otherwise it's a Nan

    for attribute in attributes_of_interest:
        data = data.dropna(subset='var_'+attribute)

    return data

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

    # checkou ttest
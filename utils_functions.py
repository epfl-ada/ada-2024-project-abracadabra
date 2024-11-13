import matplotlib.pyplot as plt
import seaborn as sns

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

    
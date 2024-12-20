import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution_grade_per_category(ratings_df, users_df,  attributes = ['appearance', 'aroma', 'palate', 'taste', 'overall']):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns='id')

    print(ratings_df.head())
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

def plot_proportion_controversial_per_category(ratings_df, users_df, beers_df, label, label_list = [2,0,1]):
    rating_user_level = ['connoisseur','enthusiast','novice']

    ratings_df = ratings_df.merge(users_df[['id','rating_user_level']], left_on = 'id_user', right_on = 'id', how='left').drop(columns=['id','appearance', 'aroma', 'palate', 'taste', 'overall'])
    ratings_df = add_label_to_comment(ratings_df=ratings_df, beers_df=beers_df, labels = label)
    #for level in rating_user_level:
    #    ratings_df_level = ratings_df[ratings_df.rating_user_level == level]
    #    plot_proportion(ratings_df_level, level, label_list = label_list)
    single_plot(ratings_df, label_list = label_list)

def add_label_to_comment(ratings_df, beers_df, labels):
    beers_df_with_label = beers_df.copy()
    beers_df_with_label['label'] = labels 

    ratings_with_labels = ratings_df.merge(beers_df_with_label[['id', 'label']], left_on='id_beer',right_on='id',how='left').drop(columns='id')

    return ratings_with_labels

def plot_proportion(ratings_df, level, label_list = [2,0,1]):
    labels = ['universal','neutral', 'controversial']
    
    label_map = {label_list[i]: labels[i] for i in range(len(labels))}
    label_counts = ratings_df['label'].value_counts().rename(label_map)
    label_frequency = label_counts/label_counts.sum()
    plt.figure(figsize=(10, 6))
    label_frequency.plot(kind='bar', color=['blue', 'orange', 'green'], alpha=0.7, edgecolor='black')
    plt.title('Number of Ratings by Label (' + level+')', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.ylabel('Number of Ratings', fontsize=14)
    plt.show()

def single_plot(ratings_df, label_list = [2,0,1]):
    label_category = ['universal','neutral', 'controversial']

    for i in range(len(label_category)):
        ratings_df.loc[ratings_df['label'] == i, 'label'] = label_category[i]

    enthusiast_proportions = ratings_df[ratings_df['rating_user_level']=='enthusiast']
    connoisseur_proportions = ratings_df[ratings_df['rating_user_level']=='connoisseur']
    novice_proportions = ratings_df[ratings_df['rating_user_level']=='novice']

    enthusiast_proportions = enthusiast_proportions.groupby('label').size()
    connoisseur_proportions = connoisseur_proportions.groupby('label').size()
    novice_proportions = novice_proportions.groupby('label').size()

    enthusiast_proportions = enthusiast_proportions/enthusiast_proportions.sum()
    connoisseur_proportions = connoisseur_proportions/connoisseur_proportions.sum()
    novice_proportions = novice_proportions/novice_proportions.sum()

    proportions_df = pd.DataFrame({
        'enthusiast': enthusiast_proportions,
        'connoisseur': connoisseur_proportions,
        'novice': novice_proportions})

    proportions_df.plot(kind='bar', figsize=(8, 6))

    # Add plot details
    plt.title('Proportion of Labels by Rating User Level')
    plt.ylabel('Proportion')
    plt.xlabel('Label')
    plt.xticks(rotation=0)
    plt.legend(title='Rating User Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


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
    user_df['rating_user_level'] = user_df['nbr_ratings_total'].apply(__classify)

    return user_df


def plot_user_expertise_distrib(df):
    """
        Plot the distribution of the expertise level of the users.
        
        Parameters:
        - df: users dataFrame
    """

    # Set up a small figure plot
    plt.figure(figsize=(6, 4))

    # Use seaborn countplot to plot the distribution
    sns.countplot(data=df, x='rating_user_level', palette='magma')

    # Add titles and labels
    plt.title(f'Distribution of users level of expertise')
    plt.xlabel('User expertise level')
    plt.ylabel('Count')

    plt.show()


def compute_rating_by_user_time(ratings_df, users_df, beers_df, label, label_list = [1,2,0], user_level = [20,800]):
    '''
    Computes the proportion of ratings made by the different class of users: [novice, enthusiast, connoisseur]. Uses the time information to classify the users through time
    
    Parameters :
    - ratings_df: Dataframe containing the ratings
    - user_df: DataFrame containing user data.
    - beers_df: DataFrame containing user data.
    - label attributed by the GMM clustering
    - label_list: list to tell us which label corresponds to which class for the 1st GMM
    - user_level: limit set to decide when a users becomes enthusiast and connoisseur default is at 20 and 800 ratings

    '''
    #We need ratings_df cause it has the dates, therefore we must also redrop all the nan
    label_category = ['neutral','controversial', 'universal']
    
    #Deep copy and keep only usefull data
    ratings_df = ratings_df[['id_user','id_beer','date']].copy(deep = True)
    users_df = users_df[['id','rating_user_level']].copy(deep = True)
    beers_df = beers_df.copy(deep = True)

    #Add labels to comment + keep only the ratings without any label
    ratings_df = add_label_to_comment(ratings_df=ratings_df, beers_df=beers_df, labels = label)
    ratings_df = ratings_df[~ratings_df.label.isna()]
    ratings_df = ratings_df[ratings_df.id_user.isin(users_df.id)]

    #Group the 1st 25 ratings for each user
    ratings_df = ratings_df.sort_values(by=['id_user', 'date'])
    ratings_grouped = ratings_df.groupby('id_user').head(user_level[0])
    label_counts = ratings_grouped.groupby('id_user')['label'].value_counts().unstack(fill_value=0)

    for label in [0, 1, 2]:
        if label not in label_counts.columns:
            label_counts[label] = 0

    #Compute the controversial, universal and neutral comments for the novice users
    label_counts = label_counts.reset_index().rename(columns={0: str(label_category[label_list[0]])+"_novice", 1: str(label_category[label_list[1]])+"_novice", 2: str(label_category[label_list[2]])+"_novice"})
    users_df = users_df.merge(label_counts, left_on='id', right_on='id_user', how='left')
    users_df["total_novice"] = users_df[str(label_category[label_list[0]])+"_novice"]+users_df[str(label_category[label_list[1]])+"_novice"]+users_df[str(label_category[label_list[2]])+"_novice"]

    #Group the 25 to 800 ratings for each user
    ratings_df = ratings_df.sort_values(by=['id_user', 'date'])
    ratings_grouped = ratings_df.groupby('id_user').apply(lambda group: group.iloc[user_level[0]:user_level[1]]).reset_index(drop=True)
    label_counts = ratings_grouped.groupby('id_user')['label'].value_counts().unstack(fill_value=0)

    for label in [0, 1, 2]:
        if label not in label_counts.columns:
            label_counts[label] = 0

    #Compute stats for the enthusiast users
    label_counts = label_counts.reset_index().rename(columns={0: str(label_category[label_list[0]])+"_enthusiast", 1: str(label_category[label_list[1]])+"_enthusiast", 2: str(label_category[label_list[2]])+"_enthusiast"})
    users_df = users_df.merge(label_counts, left_on='id', right_on='id_user', how='left')
    users_df["total_enthusiast"] = users_df[str(label_category[label_list[0]])+"_enthusiast"]+users_df[str(label_category[label_list[1]])+"_enthusiast"]+users_df[str(label_category[label_list[2]])+"_enthusiast"]

    #Group last 800 ratings for each user
    ratings_df = ratings_df.sort_values(by=['id_user', 'date'])
    ratings_grouped = ratings_df.groupby('id_user').apply(lambda group: group.iloc[user_level[1]:]).reset_index(drop=True)
    label_counts = ratings_grouped.groupby('id_user')['label'].value_counts().unstack(fill_value=0)

    for label in [0, 1, 2]:
        if label not in label_counts.columns:
            label_counts[label] = 0

    #Compute stats for the connoisseur users
    label_counts = label_counts.reset_index().rename(columns={0: str(label_category[label_list[0]])+"_connoisseur", 1: str(label_category[label_list[1]])+"_connoisseur", 2: str(label_category[label_list[2]])+"_connoisseur"})
    users_df = users_df.merge(label_counts, left_on='id', right_on='id_user', how='left')
    users_df["total_connoisseur"] = users_df[str(label_category[label_list[0]])+"_connoisseur"]+users_df[str(label_category[label_list[1]])+"_connoisseur"]+users_df[str(label_category[label_list[2]])+"_connoisseur"]

    users_df = users_df.drop(columns = ['id_user_x','id_user_y','id_user'])

    #Computes frequencies
    for label_val in label_list:
        users_df[str(label_category[label_val])+"_novice"] = users_df[str(label_category[label_val])+"_novice"]/users_df["total_novice"]
        users_df[str(label_category[label_val])+"_enthusiast"] = users_df[str(label_category[label_val])+"_enthusiast"]/users_df["total_enthusiast"]
        users_df[str(label_category[label_val])+"_connoisseur"] = users_df[str(label_category[label_val])+"_connoisseur"]/users_df["total_connoisseur"]

    plot_proportion_controversial_per_category_new(users_df)

    del ratings_df, beers_df, users_df



def plot_proportion_controversial_per_category_new(users_df):
    """
    Bar plot of the proportion of labeled comments for each class of user
    Parameters :
    - users_df: User Dataframe with the proportions we need
    """
    #Compute the mean proportion
    proportions_df = pd.DataFrame({
        'controversial': [
            users_df['controversial_novice'].mean(),
            users_df['controversial_enthusiast'].mean(),
            users_df['controversial_connoisseur'].mean()
        ],
        'neutral': [
            users_df['neutral_novice'].mean(),
            users_df['neutral_enthusiast'].mean(),
            users_df['neutral_connoisseur'].mean()
        ],
        'universal': [
            users_df['universal_novice'].mean(),
            users_df['universal_enthusiast'].mean(),
            users_df['universal_connoisseur'].mean()
        ]
    }, index=['novice', 'enthusiast', 'connoisseur'])

    #Compute the standard error for confidence intervals
    std_errors = pd.DataFrame({
        'controversial': [
            users_df['controversial_novice'].sem(),
            users_df['controversial_enthusiast'].sem(),
            users_df['controversial_connoisseur'].sem()
        ],
        'neutral': [
            users_df['neutral_novice'].sem(),
            users_df['neutral_enthusiast'].sem(),
            users_df['neutral_connoisseur'].sem()
        ],
        'universal': [
            users_df['universal_novice'].sem(),
            users_df['universal_enthusiast'].sem(),
            users_df['universal_connoisseur'].sem()
        ]
    }, index=['novice', 'enthusiast', 'connoisseur'])

    barWidth = 0.25
    r1 = np.arange(len(proportions_df)) - barWidth
    r2 = np.arange(len(proportions_df))
    r3 = np.arange(len(proportions_df)) + barWidth

    #Plot the results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(r1, proportions_df['controversial'], 
           yerr=1.96 * std_errors['controversial'], capsize=5, width=barWidth, label='Controversial')
    ax.bar(r2, proportions_df['neutral'], 
           yerr=1.96 * std_errors['neutral'], capsize=5, width=barWidth, label='Neutral')
    ax.bar(r3, proportions_df['universal'], 
           yerr=1.96 * std_errors['universal'], capsize=5, width=barWidth, label='Universal')
    ax.set_xticks([r + barWidth for r in range(len(proportions_df))])
    ax.set_xticklabels(proportions_df.index)
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Rating user level')
    ax.set_title('Proportion of labels by rating user level with 95% CI')
    ax.legend(title='Label Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

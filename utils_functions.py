def classify_user_rating_level(user_df, novice_level=20, enthusiasts_level=100):
    
    def __classify(nbr_ratings):
        if nbr_ratings <= novice_level:
            return 'novice'
        elif nbr_ratings <= enthusiasts_level:
            return 'enthusiast'
        else:
            return 'connoisseur'
        
    user_df['rating_user_level'] = user_df['nbr_ratings'].apply(__classify)

    return user_df
    
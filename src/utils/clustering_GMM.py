from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import pandas as pd
import numpy as np


def reduction_attribute(attributes):
    attributes_scaled = scale(attributes)
    attributes_scaled_reduce = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0, verbose=1).fit_transform(attributes_scaled)
    return attributes_scaled_reduce

def apply_gmm(attributes, attributes_scaled_reduce, n_components_gmm=3):
    attributes_scaled = scale(attributes)
    gmm = GaussianMixture(n_components = n_components_gmm, random_state=0)

    labels = gmm.fit_predict(attributes_scaled)

    fig, axs = plt.subplots(1, 1, figsize=(4, 4), sharey=True)
    axs.scatter(attributes_scaled_reduce[:, 0], attributes_scaled_reduce[:, 1], c=labels, alpha=0.6)
    axs.set_title("Discovered clusters with GMM")
    plt.show()
    return labels

def visualize_gmm_all_dimensions(attributes, n_components_gmm = 3):
    '''
    Visualize the clusters made by a Gaussian Mixture Model (GMM) on the data

    Parameters :
    - attributes: The variance of the different grades for each beer over the various attributes
    - n_components_gmm: number of clusters for the GMM

    Returns :
    - labels: The predicted labels of the beers from attributes. 
    '''
    #Scale, apply and predict the labels of the beers
    attributes_scaled = scale(attributes)
    gmm = GaussianMixture(n_components = n_components_gmm, random_state=0)
    labels = gmm.fit_predict(attributes_scaled)

    #Set to a Dataframe for later
    attributes_df = pd.DataFrame(attributes_scaled, columns=[f"Dim_{i+1}" for i in range(attributes_scaled.shape[1])])

    column_names = attributes.columns if hasattr(attributes, 'columns') else [f"Dim_{i+1}" for i in range(attributes_scaled.shape[1])]
    attributes_df.columns = column_names

    attributes_df['Label'] = labels  #Add labels as a column to the attributes dataframe of origin

    #Plot the results with Seaborn
    palette = sns.color_palette("deep", n_components_gmm)
    sns.pairplot(attributes_df, hue='Label', diag_kind='kde', palette=palette)
    plt.suptitle("Pairwise Plots of Scaled Attributes by GMM Labels", y=1.02)
    plt.show()

    return labels

def scale(ratings):
    ratings_scaled = StandardScaler().fit(ratings).transform(ratings)
    return ratings_scaled

def plot_nll(attributes, min=2, max=10):
    '''
    Plots the negative log-likelihood for a various number of clusters for a Gaussian Mixture Model

    Parameters :
    - attributes: The variance of the different grades for each beer over the various attributes
    - min: Min number of clusters
    - max: Max number of clusters

        '''
    nll = []
    for k in range(min, max+1):
        gmm = GaussianMixture(n_components=k, random_state=10).fit(attributes)
        nll.append({"k": k, "nll": -gmm.score(attributes) * len(attributes)})

    #Convert to Dataframe to plot more easily
    nll = pd.DataFrame(nll)
    
    # Plot the results
    plt.plot(nll.k, nll.nll, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Evaluation of the Gaussian Mixture Model for various number of clusters")
    plt.show()

def print_label_statistics(labels):
    '''
    Print the statistics of the clusters predicted from the GMM

    Parameters :
    - labels: predicted labels for the different beers 

    '''
    unique_label = np.unique(labels)
    for label in unique_label:
        print("The cluster {} contains {} beers, which represents {:.2f}% of the dataset.".format(label,len(labels[labels==label]),len(labels[labels ==label])/len(labels)*100))

def compare_two_clustering(attributes_variance, attributes_variance_sent, labels, labels_sent, label_list, label_list_sent):
    '''
    Compares the cluster predicted by GMMs applied on different data

    Parameters :
    - attributes_variance: The variance of the data used for the 1st GMM
    - attributes_variance_sent: The variance of the data used for the 2nd GMM
    - labels: labels predicted by the 1st GMM
    - labels_sent: labels predicted by the 2nd GMM
    - label_list: list to tell us which label corresponds to which class for the 1st GMM
    The list should give the label which corresponds to the following:[universal, neutral, controversial]
    - label_list_sent: list to tell us which label corresponds to which class for the 2nd GMM, same norm applies as before
    '''

    #Init list of correspondance.
    labels_name = ['universal', 'neutral', 'controversial']

    #Deep copy for security
    attributes_variance_with_label = attributes_variance.copy(deep = True)
    attributes_variance_sent_with_label = attributes_variance_sent.copy(deep = True)

    #Apply labels and drop useless columns for rest of function
    attributes_variance_with_label['label_without_sent'] = labels
    attributes_variance_with_label = attributes_variance_with_label.drop(columns=['appearance','aroma','palate','taste','overall'])
    attributes_variance_sent_with_label['label_with_sent'] = labels_sent
    attributes_variance_sent_with_label = attributes_variance_sent_with_label.drop(columns=['appearance','aroma','palate','taste','overall','sentiment_bert'])

    #Merge both datasets together on beer_id. Enables us to keep the labels of each that are in both cases
    common_beer_ids = pd.merge(attributes_variance_with_label, attributes_variance_sent_with_label, on="id_beer", how="inner")

    label_map = dict(zip(label_list, labels_name))
    label_map_sent = dict(zip(label_list_sent, labels_name))

    #Applies real name of class, and not the number anymore
    common_beer_ids['label_without_sent_name'] = common_beer_ids['label_without_sent'].map(label_map)
    common_beer_ids['label_with_sent_name'] = common_beer_ids['label_with_sent'].map(label_map_sent)
    common_beer_ids = common_beer_ids.drop(columns=['label_without_sent','label_with_sent'])
    #Computes the beers which match by their label
    percentage = common_beer_ids[common_beer_ids['label_without_sent_name'] == common_beer_ids['label_with_sent_name']]
    
    #Print various statistics to understand easier the results we have
    print("The percentage of beers which have the same label are", len(percentage)/len(common_beer_ids)*100,"%.")

    #Compares for each class in each dataset (by using the reduced one) the percentage of similarity with the other dataset
    print("\n In total in this subset of beers which contains a textual review. We have:")
    for label in labels_name:
        sub_frame = common_beer_ids[common_beer_ids.label_with_sent_name== label]
        sub_sub_frame = sub_frame[sub_frame.label_without_sent_name==label]
        print("-", len(sub_frame),"beers are labeled as", label, "when clustered with sentiment. In this we have", 100*len(sub_sub_frame)/len(sub_frame), "% of the beers without sentiment that have the same label.",)

    for label in labels_name:
        sub_frame = common_beer_ids[common_beer_ids.label_without_sent_name== label]
        sub_sub_frame = sub_frame[sub_frame.label_with_sent_name==label]
        print("-", len(sub_frame),"beers are labeled as", label, "when clustered without sentiment. In this we have", 100*len(sub_sub_frame)/len(sub_frame), "% of the beers with sentiment that have the same label.",)

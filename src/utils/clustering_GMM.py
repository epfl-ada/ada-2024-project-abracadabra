from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pandas as pd


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


def scale(ratings):
    ratings_scaled = StandardScaler().fit(ratings).transform(ratings)
    return ratings_scaled

def plot_nll(attributes, min=2, max=10):
    nll = []
    for k in range(min, max+1):
        print(k)
        # Fit the Gaussian Mixture Model
        gmm = GaussianMixture(n_components=k, random_state=10).fit(attributes)
        # Append the negative log-likelihood (lower is better)
        nll.append({"k": k, "nll": -gmm.score(attributes) * len(attributes)})

    # Convert to DataFrame
    nll = pd.DataFrame(nll)
    
    # Plot the results
    plt.plot(nll.k, nll.nll, marker="o")
    plt.xlabel("Number of Components (K)")
    plt.ylabel("Negative Log-Likelihood (NLL)")
    plt.title("Model Quality vs Number of Components")
    plt.show()
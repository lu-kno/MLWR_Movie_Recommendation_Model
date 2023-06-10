#!/usr/bin/env python
# %%

import json
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from matplotlib import cm
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class ModelTest:
    def __init__(self, file):
        # Load data from CSV file
        self.data = pd.read_csv(file)
        self.movies = self.data['movieId']
        self.tag_relevances = self.data.drop('movieId', axis=1)

        # Create results directory if not exists
        if not os.path.exists("results"):
            os.makedirs("results")

    def plot_clusters(self, column, method_name):
        print(f'running plot_clusters')
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.tag_relevances)

        # Create a scatter plot of the two principal components with cluster labels as color
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.data[column], cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'{method_name} Clustering')
        plt.savefig(f'results/{method_name}_clustering_2.png')
        plt.clf()
        
    def predict(self,*args,**kwargs):
        return self.kmeansModel.predict(*args,**kwargs)

    def kmeans_clustering(self, n=10, max_iter=100):
        print(f'running kmeans_clustering')
        # Apply KMeans Clustering
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=max_iter, n_init=1)
        kmeans.fit(self.tag_relevances)
        self.data["Category"] = kmeans.predict(self.tag_relevances)
        self.plot_clusters("Category", "KMeans")
        self.kmeansModel = kmeans

    def dbscan_clustering(self):
        print(f'running dbscan_clustering')
        # Apply DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan.fit(self.tag_relevances)
        self.data["DBSCAN_Cluster"] = dbscan.labels_
        self.plot_clusters("DBSCAN_Cluster", "DBSCAN")

    def hierarchical_clustering(self):
        print(f'running hierarchical_clustering')
        # Apply Hierarchical Clustering
        Z = linkage(self.tag_relevances, 'ward')
        self.data["Hierarchical_Cluster"] = fcluster(Z, t=5, criterion='maxclust')
        self.plot_clusters("Hierarchical_Cluster", "Hierarchical")

    def spectral_clustering(self):
        print(f'running spectral_clustering')
        # Apply Spectral Clustering
        spectral = SpectralClustering(n_clusters=5, assign_labels='discretize', random_state=0)
        spectral.fit(self.tag_relevances)
        self.data["Spectral_Cluster"] = spectral.labels_
        self.plot_clusters("Spectral_Cluster", "Spectral")

    def gmm_clustering(self):
        print(f'running gmm_clustering')
        # Apply Gaussian Mixture Models
        gmm = GaussianMixture(n_components=5, random_state=0)
        gmm.fit(self.tag_relevances)
        self.data["GMM_Cluster"] = gmm.predict(self.tag_relevances)
        self.plot_clusters("GMM_Cluster", "GMM")

    def save_results(self, file):
        print(f'running save_results')
        # Save results to a new CSV file
        self.data.to_csv(file, index=False)



categoryCount = 50

if __name__ == '__main__':
    # Usage
    model_test = ModelTest('genomeScores_usable.csv')
    model_test.kmeans_clustering(n=categoryCount)
    # model_test.dbscan_clustering()
    # model_test.hierarchical_clustering()
    # model_test.spectral_clustering()
    # model_test.gmm_clustering()
    model_test.save_results('movies_clustered_kmeans.csv')
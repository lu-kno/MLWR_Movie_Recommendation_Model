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



# %%

genomeScores = pd.read_csv('genomeScores_usable.csv')
# %%

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
    
    
# %%
data = model_test.data.set_index('movieId')
groups = data.groupby('Category')

res=dict()

for g, group in groups:
    group.drop(columns=['Category'], inplace=True)
    pd.concat([group,group.mean()],)
    group.sort_values(by=group.index[-1], axis=1, ascending=False, inplace=True)
    plt.figure()
    plt.imshow(group)
    plt.savefig(f'results/ClusterFilms_{g}.png')
    plt.close()
    res[str(group.columns[0:3])]=group.index.tolist()
    group.to_csv(f'results/ClusterFilms_{g}.csv')

json.dump(res, open('ClustersFound.json', 'w+'), indent=4)
    
    

# Get list of films from user
# Categorize films from user
# --> This describes movie taste of user and will be used for recomendations

# model with 100 categories
# model with 30 categories

# %%

class User():
    def __init__(self, filmlist, model=None) -> None:        
        self.preferences: np.ndarray[float] = []    
        self.userFilmList: pd.DataFrame = pd.DataFrame(filmlist,columns=['movieId','rating'])
        self.userFilmList.loc[:,'Category'] = None
        
        if model:
            self.calcPreferences()
    
    def calcPreferences(self, model=model_test):
        # cats = model_test.data.loc[model_test.data['col1'].isin(self.userFilmList['movieId'])]
        
        allFilms = model.data[['movieId', 'Category']]
        
        self.userFilmList.loc[:,'Category'] = self.userFilmList.apply(lambda row: allFilms.loc[allFilms['movieId']==row['movieId'],'Category'].values[0], axis=1)
        
        
        categoryUserRatingSorted = self.userFilmList.drop(columns='movieId').groupby('Category').mean().sort_values(by='Category', ascending=True)
        
        preferencesRaw = np.zeros(categoryCount)
        
        for ind, row in categoryUserRatingSorted.iterrows():
            preferencesRaw[ind] = row['rating']
            
        self.preferences = preferencesRaw/np.sum(preferencesRaw)
        
        return self.preferences
    
    

def pickFilmFromPreferences(preferences = None):
    if preferences is None:
        preferences = np.ones(categoryCount)/categoryCount

    rnumber = random.random()
    
    preferencesCumulativeSum = np.cumsum(preferences)
    
    for catId, pref in enumerate(preferencesCumulativeSum):
        if rnumber<pref:
            return pickFilmFromCategory(catId)

    return None


def pickFilmFromCategory(catId):
    return (catId,random.sample(sorted(model_test.data.loc[model_test.data['Category']==catId, 'movieId']),1))
    
    
    # pick film from category
    # return film


# %%

# open csv with user ratings
ratings_df = pd.read_csv('./ml-25m/ratings.csv')

movieIds = pd.read_csv('ml-25m/movies.csv')
movieIdsDict = movieIds.set_index('movieId').to_dict()['title']

user_groups = ratings_df.groupby('userId')

users: List[User] = []

for user_id, user_ratings in user_groups:
    if user_id > 5:
        break
    user_ratings = user_ratings.set_index('movieId').drop(columns=['userId','timestamp'])
    user_ratings = user_ratings.rename(index=movieIdsDict).reset_index()
    users.append(User(user_ratings,model=model_test))


# %%

print(pickFilmFromPreferences(preferences = users[0].preferences))
    
# # %%
# means = groups.mean()
# # %%
# means.to_csv('cluster_means_kmeans.csv')

# %%
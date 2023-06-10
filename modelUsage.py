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

genomeScores = pd.read_csv('Data/genomeScores_usable.csv')
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

    def save_results(self, file):
        print(f'running save_results')
        # Save results to a new CSV file
        self.data.to_csv(file, index=False)


CATEGORY_COUNT = 50

model_test = ModelTest('Data/genomeScores_usable.csv')
model_test.kmeans_clustering(n=CATEGORY_COUNT)
model_test.save_results('Data/movies_clustered_kmeans.csv')
    
    
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
        
        preferencesRaw = np.zeros(CATEGORY_COUNT)
        
        for ind, row in categoryUserRatingSorted.iterrows():
            preferencesRaw[ind] = row['rating']
            
        self.preferences = preferencesRaw/np.sum(preferencesRaw)
        
        return self.preferences
    
    

def pickFilmFromPreferences(preferences = None):
    if preferences is None:
        preferences = np.ones(CATEGORY_COUNT)/CATEGORY_COUNT

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
#!/usr/bin/env python
# %%

import json
import os
import random
from typing import (Any, Dict, Hashable, Iterable, List, Optional, Sequence,
                    Set, Tuple, Union)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

os.mkdir('results')

# %%

genomeScores = pd.read_csv('Data/prep_genomeScores_usable.csv')
# %%

class KMCModel:
    def __init__(self, file):
        # Load data from CSV file
        self.data = pd.read_csv(file)
        self.movies = self.data['movieId']
        self.tag_relevances = self.data.drop('movieId', axis=1)
        self.n=-1

        # Create results directory if not exists
        if not os.path.exists("results"):
            os.makedirs("results")

    def plot_clusters(self, column, dotsne=False):
        print(f'running plot_clusters')
        # Reduce dimensionality to 2D using PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.tag_relevances)

        # Create a scatter plot of the two principal components with cluster labels as color
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.data[column], cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'kmeans_clustering Clustering PCA Plot K={self.n}')
        plt.savefig(f'results/kmeans_clustering_clustering_PCA_2_K{self.n}.png')
        plt.clf()
        
        if dotsne:
            # t-SNE plot
            tsne = TSNE(n_components=2, verbose=1, perplexity=self.n, n_iter=300)
            tsne_results = tsne.fit_transform(self.tag_relevances)
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=self.data[column], cmap='viridis')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f'kmeans_clustering Clustering t-SNE Plot K={self.n}')
            plt.savefig(f'results/kmeans_clustering_clustering_tSNE_2_K{self.n}.png')
            plt.clf()
            
        
    def predict(self,*args,**kwargs):
        return self.kmeansModel.predict(*args,**kwargs)
    
    def getFilmCategory(self, filmId):
        if filmId not in self.movies.values:
            return None
        # return self.data.loc[self.data['movieId']==filmId,'Category'].values[0]
        return self.data.loc[self.data['movieId']==filmId,'Category'].values[0]

    def kmeans_clustering(self, n=10, max_iter=100):
        self.n = n
        print(f'running kmeans_clustering')
        # Apply KMeans Clustering
        kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=max_iter, n_init=1)
        kmeans.fit(self.tag_relevances)
        self.data["Category"] = kmeans.predict(self.tag_relevances)
        self.plot_clusters("Category",)
        self.kmeansModel = kmeans

    def save_results(self, file):
        print(f'running save_results')
        # Save results to a new CSV file
        self.data.to_csv(file, index=False)


# for i in [5,6,7,8,9]:
#     model_test_n = KMCModel('Data/genomeScores_usable.csv')
#     model_test_n.kmeans_clustering(n=i)
    
    

CATEGORY_COUNT = 50
model_test = KMCModel('Data/prep_genomeScores_usable.csv')
model_test.kmeans_clustering(n=CATEGORY_COUNT)
model_test.save_results('Data/rmodel_movies_clustered_kmeans.csv')
model_test.plot_clusters("Category")
    
    
# %%

data = model_test.data.set_index('movieId')
groups = data.groupby('Category')

res=dict()

for categoryInd, categoryGroup in groups:
    categoryGroup.drop(columns=['Category'], inplace=True)
    categoryGroup.loc['mean'] = categoryGroup.mean()
    categoryGroup.sort_values(by=categoryGroup.index[-1], axis=1, ascending=False, inplace=True)
    sumOfRelevances = categoryGroup.sum(axis=0)
    maxRelevance = sumOfRelevances.max()
    tagMaxRelevance = sumOfRelevances.idxmax()
    plt.figure()
    plt.imshow(categoryGroup)
    plt.savefig(f'results/ClusterFilms_{categoryInd}{tagMaxRelevance}.png')
    plt.close()
    res[str(categoryGroup.columns[0:5])]=categoryGroup.index.tolist()
    categoryGroup.to_csv(f'results/ClusterFilms_{categoryInd}{tagMaxRelevance}.csv')


with open('Data/rmodel_ClustersFound.json', 'w+') as f:
    json.dump(res, f, indent=4)
    

# Get list of films from user
# Categorize films from user
# --> This describes movie taste of user and will be used for recomendations

# model with 100 categories
# model with 30 categories

# %%

class User():
    def __init__(self, filmlist: Sequence, model: Optional[KMCModel] = None) -> None:        
        self.preferences: np.ndarray = np.array([])    
        self.userFilmList: pd.DataFrame = pd.DataFrame(filmlist,columns=['movieId','rating'])
        self.userFilmList.loc[:,'Category'] = None
        
        if model:
            self.getCategories(model=model)
            self.calcPreferences(model=model)
    
    def getCategories(self,model: KMCModel = model_test) -> None:
        self.userFilmList.loc[:,'Category'] = self.userFilmList['movieId'].apply(model.getFilmCategory)

    
    def calcPreferences(self, model: KMCModel = model_test) -> np.ndarray:
        # cats = model_test.data.loc[model_test.data['col1'].isin(self.userFilmList['movieId'])]
        
        # allFilms = model.data[['movieId', 'Category']]

        
        # self.userFilmList.loc[:,'Category'] = self.userFilmList.apply(lambda row: allFilms.loc[allFilms['movieId']==row['movieId'],'Category'].values[0], axis=1)
        
        
        categoryUserRatingSorted = self.userFilmList.drop(columns='movieId').groupby('Category').mean().sort_values(by='Category', ascending=True)
        
        preferencesRaw = np.zeros(CATEGORY_COUNT)
        
        for ind, row in categoryUserRatingSorted.iterrows():
            preferencesRaw[int(ind)] = row['rating']
            
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

# %%
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
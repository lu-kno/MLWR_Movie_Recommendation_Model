#!/usr/bin/env python
# %%

import json
import os
from typing import (Any, Dict, Hashable, Iterable, List, Optional, Sequence,
                    Set, Tuple, Union)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image
from scipy.stats import gaussian_kde

if not os.path.exists('Data'):
    os.mkdir('Data')


# %%
def find_relevant(df: pd.DataFrame, threshold: float) -> pd.Index:
    '''
    Returns a list of relevant tags. 
    Relevant tags are those with a mean relevance (over all films) above the threshold.
    Input:
        df: dataframe with genome scores
        threshold: float between 0 and 1
    Output:
        relevantTags: list of relevant tags (type: pd.Index)
    '''
    # get tags with mean relevance above threshold (df with tags as columns and movies as rows)
    tagMeans  = df.mean(axis=0)
    relevantTags = df.columns[tagMeans > threshold]
    return relevantTags
    
def get_relevant(df: pd.DataFrame, relevantTags: Union[list,pd.Index]) -> pd.DataFrame:
    '''
    Returns a dataframe with only the relevant tags.
    Input:
        df: dataframe with genome scores
        relevantTags: list of relevant tags (type: pd.Index or list)
    Output:
        df_rel: dataframe with only the relevant tags
    '''
    df_rel = df[relevantTags]
    # print(f'Relevant Gnome Scores Shape = {df_rel.shape}')
    return df_rel


def get_irrelevant(df: pd.DataFrame, relevantTags: Union[list,pd.Index]) -> pd.DataFrame:
    '''
    Returns a dataframe with only the relevant tags.
    Input:
        df: dataframe with genome scores
        relevantTags: list of relevant tags (Tags to drop) (type: pd.Index or list)
    Output:
        df_irrel: dataframe with only the relevant tags
    '''
    df_irrel = df.drop(columns =relevantTags)
    # print(f'Relevant Gnome Scores Shape = {df_rel.shape}')
    return df_irrel


def find_synonyms(df: pd.DataFrame, threshold: float, corr_given: bool=False, plot_corr: bool=False, corr_saveto: Optional[str] = 'Data/prep_corr_arr_relevant') -> list:
    '''
    Returns a list of synonyms.
    Input:
        df: dataframe with genome scores or correlation matrix if corr_given=True
        threshold: float between 0 and 1
        corr_given: if True, df is a correlation matrix
        plot_corr: if True, plots the density of the correlation matrix and the density of the tag means
        corr_saveto: if not None, saves the correlation matrix to the given path
    Output:
        synonyms: list of synonyms
    '''

    tagMeans  = genomeScores.mean(axis=0)
    
    # Get Correlation Matrix
    if corr_given:
        correlation_matrix = df.copy()
    else:
        correlation_matrix = df.corr()
        
    if corr_saveto is not None:
        correlation_matrix.to_csv(f'{corr_saveto}.csv')
        
    if plot_corr:   
        densitymtv = gaussian_kde(tagMeans)
        xsmtv = np.linspace(0, 1, 100)
        plt.plot(xsmtv, densitymtv(xsmtv))
        density = gaussian_kde(correlation_matrix.values.flatten())
        xs = np.linspace(-1, 1, 100)
        plt.plot(xs, density(xs))
            
    # get tags with correlation above threshold
    correlatedTagsIndices = np.argwhere(correlation_matrix.values > threshold)

    # get correlated tag pairs
    tags = correlation_matrix.columns
    correlatedTagPairs = set()
    for tag1_i,tag2_i in correlatedTagsIndices:
        if tag1_i != tag2_i:     
            # Add the tag with the larger mean relevance first
            # print(f'means: \n\t{cols[tag1_i]}={meanTags[cols[tag1_i]]} \n\t{cols[tag2_i]}={meanTags[cols[tag2_i]]}')
            if tagMeans[tags[tag1_i]] < tagMeans[tags[tag2_i]]:
                correlatedTagPairs.add(tuple([tags[tag2_i], tags[tag1_i]]))
            else:
                correlatedTagPairs.add(tuple([tags[tag1_i], tags[tag2_i]]))   

    # create list of synonyms with the first tag being the one with the largest mean relevance
    synonyms = []
    for bestTagName,unusedTagName in correlatedTagPairs:
        found = False
        for synonymSet in synonyms:
            if synonymSet[0]==unusedTagName:
                synonymSet.insert(0,bestTagName)
                found = True
                break
            if synonymSet[0]==bestTagName:
                synonymSet.append(unusedTagName)
                found = True
                break
            continue
        if not found:
            synonyms.append([bestTagName,unusedTagName])
            
    return synonyms

            
def replace_synonyms(df_: pd.DataFrame, synonyms: List[List[str]]) -> pd.DataFrame:
    '''
    Returns a dataframe with the synonym tags merged together.
    Input:
        df_: dataframe with genome scores
        synonyms: list of synonyms
    Output:
        df: dataframe with reduced number of tags
    '''
    unusable_tags = set()
    df = df_.copy()
    synonymSet: List[str] = []
    try:
        for synonymSet in synonyms:
            bestTag = synonymSet[0]
            unusable_tags.update(synonymSet[1:])
            # use most relevant tag value for each movie
            df.loc[:,bestTag] = np.max(df.loc[:,synonymSet],axis=1)
        
        df = df.drop(columns=list(unusable_tags))
    except Exception as e:
        print(f'synonymSet = {synonymSet}')
        print(f'df.columns = {df.columns}')
        raise(e)
    return df




# %%
print('Loading Data...')
print('Loading Genome Scores...')
# load csv in pandas dataframe
genomeScoresSparse = pd.read_csv('ml-25m/genome-scores.csv')
genomeScores = genomeScoresSparse.pivot(index='movieId', columns='tagId', values='relevance')

print('Loading Genome Tags...')
genomeTags = pd.read_csv('ml-25m/genome-tags.csv')
tagsDict = genomeTags.set_index('tagId').to_dict()['tag']
genomeScores = genomeScores.rename(columns=tagsDict)

print('Loading Movie Titles...')
movieIds = pd.read_csv('ml-25m/movies.csv')
movieIdsDict = movieIds.set_index('movieId').to_dict()['title']
genomeScores = genomeScores.rename(index=movieIdsDict) 

tagCount_start = len(genomeScores.columns)
# %%


MEAN_RELEVANCE_THRESHOLD = 0.08
CORR_THRESHOLD = 0.5

print('Finding Relevant Tags...')
# get relevant tags
relevantTags = find_relevant(genomeScores, MEAN_RELEVANCE_THRESHOLD)
genomeScores_relevant = get_relevant(genomeScores, relevantTags)
with open('Data/prep_relevantTags.json','w+') as f:
    json.dump(genomeScores_relevant.columns.to_list(), f)
tagCount_relevant = len(relevantTags)

# get irrelevant tags (just for reference)
genomeScores_irrelevant = get_relevant(genomeScores, relevantTags)
with open('Data/prep_irrelevantTags.json','w+') as f:
    json.dump(genomeScores_irrelevant.columns.to_list(), f)
del genomeScores_irrelevant

print('Finding Synonyms...')
# deal with synonyms
synonyms = find_synonyms(genomeScores_relevant, CORR_THRESHOLD)
genomeScores_usable = replace_synonyms(genomeScores_relevant, synonyms)
with open('Data/prep_synonyms.json','w+') as f:
    f.write('[\n')
    for syn in synonyms:
        f.write((f'{syn},\n').replace("'",'"'))
    f.write(']')
genomeScores_usable.to_csv('Data/prep_genomeScores_usable.csv')
tagCount_reduced = len(genomeScores_usable.columns)


# print(f'genome Table = {np.shape(genomeScores_usable)}')

for syn in synonyms:
    print(syn)
    
print(f'Number of tags: {tagCount_start} -> {tagCount_relevant} -> {tagCount_reduced}')




# %%


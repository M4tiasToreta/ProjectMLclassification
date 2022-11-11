import json
from re import X
from this import d
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sb
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# Reading data
def query_datadet_model():
    dataset_model = pd.read_csv("Dataset_model.csv")
    dataset_model = dataset_model[dataset_model['nullified']==0]
    dataset_model = dataset_model[dataset_model['outdated']==0]
    dataset_model = dataset_model.reset_index(drop=True)
    return dataset_model

def query_subjects_questions():
    subjects_questions = pd.read_csv("subjects_questions.csv")
    return subjects_questions

def query_submit():
    submit = pd.read_csv('Submit.csv')
    return submit

# Feature engineerig
def discipline_stats(dataset_model):
    reduced = dataset_model[['discipline_id', 'acertou']]
    difficulty = reduced.groupby(['discipline_id']).mean()
    many = reduced.groupby(['discipline_id']).count()
    return difficulty

def institution_stats(dataset_model):
    reduced = dataset_model[['institute_id', 'acertou']]
    difficulty = reduced.groupby(['institute_id']).mean()
    difficulty['acertou'] = difficulty['acertou'].apply(lambda x: 1-x)
    return difficulty

def region_stats(dataset_model):
    reduced = dataset_model[['region', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('region').mean()
    stats = stats.reset_index()
    return stats

def country_stats(dataset_model):
    reduced = dataset_model[['country', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('country').mean()
    stats = stats.reset_index()
    return stats

def device_stats(dataset_model):
    reduced = dataset_model[['device_type', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('device_type').mean()
    stats = stats.reset_index()
    return stats

def difficulty_stats(dataset_model):
    reduced = dataset_model[['difficulty', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('difficulty').mean()
    stats = stats.reset_index()
    return stats

def modality_stats(dataset_model):
    reduced = dataset_model[['modality_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('modality_id').mean()
    stats = stats.reset_index()
    return stats

def scholarity_id(dataset_model):
    reduced = dataset_model[['scholarity_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('scholarity_id').mean()
    stats = stats.reset_index()
    return stats

def product_stats(dataset_model):
    reduced = dataset_model[['product_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('product_id').mean()
    stats = stats.reset_index()
    return stats

def knowledge_area_stats(dataset_model):
    reduced = dataset_model[['knowledge_area_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('knowledge_area_id').mean()
    stats = stats.reset_index()
    return stats

def examining_board_stats(dataset_model):
    reduced = dataset_model[['examining_board_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('examining_board_id').mean()
    stats = stats.reset_index()
    return stats

def novo_user_stats(dataset_model):
    reduced = dataset_model[['novo_user_id', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('novo_user_id').mean()
    stats = stats.reset_index()
    return stats

def carrers_stats(dataset_model):
    reduced = dataset_model[['gp:carrers', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('gp:carrers').mean()
    stats = stats.reset_index()
    return stats

def segment_stats(dataset_model):
    reduced = dataset_model[['gp:segment', 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby('gp:segment').mean()
    stats = stats.reset_index()
    return stats

def mean_encoding(variable):
    reduced = dataset_model[[variable, 'acertou']]
    reduced = reduced.dropna()
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby(variable).mean()
    stats = stats.reset_index()
    return stats

# Graph_plots
def plot_graphs():

    return

# Modeling
dataset_model = query_datadet_model()
df = query_datadet_model()
df1 = dataset_model[['acertou']]
df1 = df1.reset_index()
df = df[[ 'device_type', 'country', 'region', 'institute_id', 'discipline_id', 'difficulty',
            'modality_id']]
# tirei: 
df_pivot = discipline_stats(dataset_model)
df_pivot = df_pivot.reset_index()
df = df.merge(df_pivot, how='left', on='discipline_id')
df = df.drop(columns=['discipline_id'])
df = df.rename(columns={'acertou': 'discipline'})

df_pivot = institution_stats(dataset_model)
df_pivot = df_pivot.reset_index()
df = df.merge(df_pivot, how='left', on='institute_id')
df = df.drop(columns=['institute_id'])
df = df.rename(columns={'acertou': 'institute'})

df_pivot = difficulty_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='difficulty')
df = df.drop(columns=['difficulty'])
df = df.rename(columns={'acertou': 'difficulty'})

df_pivot = region_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='region')
df = df.drop(columns=['region'])
df = df.rename(columns={'acertou': 'region'})

df_pivot = country_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='country')
df = df.drop(columns=['country'])
df = df.rename(columns={'acertou': 'country'})

df_pivot = device_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='device_type')
df = df.drop(columns=['device_type'])
df = df.rename(columns={'acertou': 'device'})

df_pivot = modality_stats(dataset_model)
df_final1 = df.merge(df_pivot, how='left', on='modality_id')
df_final1 = df_final1.drop(columns=['modality_id'])
df_final1 = df_final1.rename(columns={'acertou': 'modality'})

df = dataset_model.copy()
df = df[['scholarity_id', 'product_id', 'knowledge_area_id', 'novo_user_id', 'gp:carrers',
            'gp:segment', 'examining_board_id']]

df_pivot = scholarity_id(dataset_model)
df = df.merge(df_pivot, how='left', on='scholarity_id')
df = df.drop(columns=['scholarity_id'])
df = df.rename(columns={'acertou': 'scholaraty'})

df_pivot = product_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='product_id')
df = df.drop(columns=['product_id'])
df = df.rename(columns={'acertou': 'product'})

df_pivot = knowledge_area_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='knowledge_area_id')
df = df.drop(columns=['knowledge_area_id'])
df = df.rename(columns={'acertou': 'knowledge_area'})

df_pivot = novo_user_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='novo_user_id')
df = df.drop(columns=['novo_user_id'])
df = df.rename(columns={'acertou': 'novo_user'})

df_pivot = carrers_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='gp:carrers')
df = df.drop(columns=['gp:carrers'])
df = df.rename(columns={'acertou': 'carrers'})

df_pivot = segment_stats(dataset_model)
df = df.merge(df_pivot, how='left', on='gp:segment')
df = df.drop(columns=['gp:segment'])
df = df.rename(columns={'acertou': 'segment'})

df_pivot = examining_board_stats(dataset_model)
df_final2 = df.merge(df_pivot, how='left', on='examining_board_id')
df_final2 = df_final2.drop(columns=['examining_board_id'])
df_final2 = df_final2.rename(columns={'acertou': 'examining_board'})

df_final1 = df_final1.reset_index()
df_final2 = df_final2.reset_index()
df_final = df_final1.merge(df_final2, how='inner', on='index')

df1 = df_final.merge(df1, how='left', on='index')
df1 = df1.drop(columns=['index'])
df1 = df1.dropna()
df1 = df1.reset_index(drop=True)

x_train, x_test, y_train, y_test =  train_test_split(df1.drop('acertou', axis=1), df1['acertou'], random_state=42)

pca = PCA(n_components=7)
transformed = pca.fit_transform(x_train)
x_train = pd.DataFrame(transformed)
transformed = pca.fit_transform(x_test)
x_test = pd.DataFrame(transformed)

LogReg = LogisticRegression()
LogReg = LogReg.fit(x_train.values, y_train.values)
print(f1_score(y_test, LogReg.predict(x_test)))

LogReg.predict(x_test)




#random forest
clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
clf.fit(x_train.values, y_train.values)
clf.predict(x_test)

# f1_score
f1_score(y_test, clf.predict(x_test))

# cluster
kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
kmeans.predict(x_test)




centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

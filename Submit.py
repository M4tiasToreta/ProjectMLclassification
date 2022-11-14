import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import math

# querying


def query_datadet_model():
    dataset_model = pd.read_csv("Dataset_model.csv")
    return dataset_model


def query_submit(names):
    submit = pd.read_csv('Submit.csv')
    dataset_model = query_datadet_model()
    df = submit[names]
    for name in names:
        print(name)
        encoded = mean_encoding(name, dataset_model)
        df[name] = df[name].replace(to_replace=float('nan'), value='bolo')
        df = df.merge(encoded, how='left', on=name)
        df = df.drop(columns=[name])
        df = df.rename(columns={'acertou': name})
        df[name] = df[name].replace(to_replace=float('nan'), value=encoded.iloc[-1][1])
    return df

# feature engineering


def mean_encoding(variable, dataset_model):
    reduced = dataset_model[[variable, 'acertou']]
    reduced[variable] = reduced[variable].replace(
        to_replace=float('nan'), value='bolo')
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby(variable).mean()
    stats = stats.reset_index()
    return stats


def create_df(names):
    dataset_model = query_datadet_model()
    df1 = dataset_model[['acertou']]
    df1 = df1.reset_index()
    df = dataset_model[names]
    for name in names:
        encoded = mean_encoding(name, dataset_model)
        df[name] = df[name].replace(to_replace=float('nan'), value='bolo')
        df = df.merge(encoded, how='left', on=name)
        df = df.drop(columns=[name])
        df = df.rename(columns={'acertou': name})
    df = df.reset_index()
    df = df1.merge(df, how='left', on='index')
    df = df.drop(columns=['index'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


# Modeling
names = ['institute_id', 'difficulty',
         'novo_user_id', 'examining_board_id',
         'gp:carrers', 'nullified',
         'outdated', 'publication_year']
# tirei : 'country', 'device_type', 'region',
# 'scholarity_id' , 'product_id', 'knowledge_area_id',
#  'gp:segment' ,'examining_board_id'


df = create_df(names)
acertou = df['acertou']
df = df.drop(columns=['acertou'])

submit = query_submit(names)

lista_rf = list()


pca = PCA(n_components=4)
transformed = pca.fit_transform(df)
df = pd.DataFrame(transformed)
transformed = pca.fit_transform(submit)
submit = pd.DataFrame(transformed)

clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1, criterion='gini', n_estimators=100, bootstrap=False)
clf.fit(df.values, acertou)
y = clf.predict(submit)
submit['acertou'] = y

submit = submit['acertou']
submit.to_csv('Submit3.csv', index=False, header=False)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
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
import math
import time

# querying
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

# feature engineering
def mean_encoding(variable, dataset_model):
    reduced = dataset_model[[variable, 'acertou']]
    reduced = reduced.dropna()
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
        df = df.merge(encoded, how='left', on=name)
        df = df.drop(columns=[name])
        df = df.rename(columns={'acertou': name})
    df = df.reset_index()
    df = df1.merge(df, how='left', on='index')
    df = df.drop(columns=['index'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))

# graphs

def feature_importance_logreg(model,feature_names, df):
    w0 = model.intercept_[0]
    w = model.coef_[0]
    idx = 99
    x = df.iloc[idx][feature_names].values
    result = 0
    result += w0
    for i in range(0, 4):
        result += x[i] * w[i]
    result = sigmoid(result)
    feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
    feature_importance["importance"] = pow(math.e, w)
    feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
    ax = feature_importance.plot.barh(x='feature', y='importance')
    plt.show()

def feature_importance_random_forest(forest, feature_names):
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

# modeling


########################### Logistic Regression #############################
names = [ 'device_type', 'country', 'region', 'institute_id', 'difficulty',
          'scholarity_id', 'novo_user_id', 'gp:carrers',
          'gp:segment']

#tirei : 'product_id', 'knowledge_area_id', 'discipline_id', 'modality_id', 'examining_board_id'

df = create_df(names)


x_train, x_test, y_train, y_test =  train_test_split(df.drop('acertou', axis=1), df['acertou'], random_state=42)


LogReg = LogisticRegression()
LogReg.fit(x_train.values, y_train.values)
feature_importance_logreg(LogReg, names, df)
print('------------- logreg --------------')
print(f1_score(y_test, LogReg.predict(x_test)))


########################### Random Forest #############################


names = [ 'institute_id', 'difficulty',
          'novo_user_id', 'gp:carrers',
          'examining_board_id']
# tirei : 'country', 'device_type', 'region', 
# 'scholarity_id' , 'product_id', 'knowledge_area_id', 'gp:segment', 'discipline_id'

df = create_df(names)


x_train, x_test, y_train, y_test =  train_test_split(df.drop('acertou', axis=1), df['acertou'], random_state=87)

pca = PCA(n_components=4)
transformed = pca.fit_transform(x_train)
x_train = pd.DataFrame(transformed)
transformed = pca.fit_transform(x_test)
x_test = pd.DataFrame(transformed)

clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
clf.fit(x_train.values, y_train.values)
# feature_importance_random_forest(clf, names)
print('------------- randomForest --------------')
print(f1_score(y_test, clf.predict(x_test)))






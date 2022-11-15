# ProjectMLclassification

### Summary

* About the competition and datasets
* Libraries used throughout this document
* Data preprocessing with mean encoding
* Modeling
  * Random Forest
  * Logistic Regression
  * XGBClassification
* Feature importance
  * Random Forest
  * Logistic Regression
  * XGBClassification
* Validation
* Cross validation
* PCA dimension decomposition
* Final conclusions
* References

### About the competition and datasets

The competition gave us two main datasets and a txt document explaining the columns. The first dataset had 100 questions for each one of 20.000 students, wich have been answered and had a column that indicates whether they got it right or wrong, called "acertou"(means got it right in portuguese). Summing up 2.000.000 rows, that's the dataset our model is supposed to learn from. The second dataset is called Submit, it's the 
dataset the model is supposed to predict. This second dataset has 20.000 rows, one for each student with a question.

If you want to checkout the data used in this project, all the datasets and txt file with columns descriptions have been listed in the drive below: https://drive.google.com/drive/folders/1VtFkocuaNB8oo3bqF0VCx6ctnJwXitx_

### Libraries used throughout this document

The following libraries(along with many others used while the creative process of the competition) can also be seen in the 
main.py or tests.py python archives in this repository.

```
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
import math
import time
from xgboost import XGBClassifier
from matplotlib import pyplot
```

### Data preprocessing with mean encoding

Most of data in the dataset is categorical, what makes it impossible for machine learning models to calculate previsions. So, I decided to apply mean encoding method which calculates the average of correct answers each label has, this way models can calculate with numerical data. For that, I made the following generic function, in which you only have to input the name of the column you want to encode and the dataset you are getting data from.

```
def mean_encoding(variable, dataset_model):
    reduced = dataset_model[[variable, 'acertou']]
    reduced[variable] = reduced[variable].replace(
        to_replace=float('nan'), value='bolo')
    reduced = reduced.reset_index(drop=True)
    stats = reduced.groupby(variable).mean()
    stats = stats.reset_index()
    return stats
```

Because every single categorical data is going to be processed by this function, I had to turn NaN(Not a number or null) cells into strings, so it could
be calculated in the mean encoding. For that, I made this function when querying the submit dataset:

```
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
```

You might notice that this function is not covered for the dataset model, the dataset the model is actually going to learn from. The reason for this is that the NaN treatment for the dataset_model is made in the function where creates the training dataframe to go straight to the model:


```
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
```


```
def query_datadet_model():
    dataset_model = pd.read_csv("Dataset_model.csv")
    return dataset_model
```

The second function above is just so the first one is not incomplete, it is for reading the main dataset and turning it to a dataframe. The first one
is the creating dataframe which will be used in the model. This function only requires inputing names of columns in a list type variable for those columns
you want your model to calculate with.

### Modeling

After preprocessing the data, it's time to start modeling different MLs to check and use the most accurate one. In this document, I will focus in only three,
although more were tested. Documented models are: Random forest, Logistic Regression and XGBClassification.

#### Random Forest

Random forest is a machine learning model based in a forest of decision trees that, with the samples given, calculates the mean of decisions of each tree
and consider those to the final answer.

The first tests with random forest, from the scikit learn library, were implemented using the following code (notice that we are passing the names list with the columns names, so we can filter dataset_model and mean encode it):

```
names = ['institute_id', 'difficulty',
         'novo_user_id', 'examining_board_id',
         'gp:carrers', 'nullified',
         'outdated', 'publication_year',
         'country', 'device_type', 'region',
         'scholarity_id' , 'product_id', 'knowledge_area_id',
         'gp:segment' ,'examining_board_id']
         
df = create_df(names)

x_train, x_test, y_train, y_test = train_test_split(df.drop('acertou', axis=1), df['acertou'])
clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
clf.fit(x_train.values, y_train.values)
```

The train_test_split is a way to split your data so you can train your model and evaluate it according to the results you want. For now, we are going to
leave like this and move on, but later on we are going to talk about validation, cross validation, feature importance and how to make this model more
accurate.

#### Logistic Regression

Logistic Regression is a statistics formula in which its base function depends on variables and cofficients.

y = B0*x0 + B1*x1 + B2*x2 ... Bn*Xn

and the binary probability comes from the sigmoid:

p = 1/(1 + e**-y)

Also used from scikit learn. The model was first tested with this script:

```
names = ['institute_id', 'difficulty',
         'novo_user_id','gp:carrers',
         'nullified', 'outdated',
         'product_id', 'knowledge_area_id',
         'discipline_id', 'modality_id',
         'examining_board_id', 'country']
         
df = create_df(names)

x_train, x_test, y_train, y_test = train_test_split(df.drop('acertou', axis=1), df['acertou'])

LogReg = LogisticRegression()
LogReg.fit(x_train.values, y_train.values)

LogReg.predict(x_test)
```

In feature importance and in validation this models will be formally tested, for now we are simply applying and returning the array with 0's and 1's predicted.

#### XGBClassification

 "The XGBoost stands for eXtreme Gradient Boosting, which is a boosting algorithm based on gradient boosted decision trees algorithm.
 XGBoost applies a better regularization technique to reduce overfitting, and it is one of the differences from the gradient boosting."
 This is the definition made by datatechnotes. Basically, XGBClassification, is a type of decision tree.
 
It was used through its own library. Code used for first trains:
 
 ```
names = ['institute_id', 'difficulty',
         'novo_user_id','gp:carrers',
         'nullified', 'outdated',
         'product_id', 'knowledge_area_id',
         'discipline_id', 'modality_id',
         'examining_board_id', 'country']
         
df = create_df(names)

x_train, x_test, y_train, y_test = train_test_split(df.drop('acertou', axis=1), df['acertou'])

bgc = XGBClassifier()
bgc.fit(x_train, y_train)

bgc.predict(x_test)
```

### Feature importance

Feature importance is a graphic way to visualize the importance of the dimentions on your model, so we can better fit it. Each model has its own way to
show feature importance. Following:

#### Random forest
  
Inspired by scikit learn docs, I created a function in main.py that plots the model's feature importance and integrated it in random forest scrypt:

```
def feature_importance_random_forest(forest, feature_names):
    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    print(
        f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
 ```

 ```
names = ['institute_id', 'difficulty',
         'novo_user_id', 'examining_board_id',
         'gp:carrers', 'nullified',
         'outdated', 'publication_year',
         'country', 'device_type', 'region',
         'scholarity_id' , 'product_id', 'knowledge_area_id',
         'gp:segment' ,'examining_board_id']
         
df = create_df(names)

x_train, x_test, y_train, y_test = train_test_split(df.drop('acertou', axis=1), df['acertou'])
clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1)
clf.fit(x_train.values, y_train.values)

feature_importance_random_forest(clf, names)
 ```
 
 This function requires names of the columns the model and the model fitted itself to plot the feature importances. This way we'll be able only to
 fit the model with the most important variables.
 
 #### Logistic Regression
 
 As discussed before, logistic regression is a function and the importance of each dimension is its coefficient, so if we want to plot it, we need to
 make python understand each coefficient and plot it. I also made a generic function so it would be easier for anyone who finds this article to
 replicate it(much scalable too).
 
 ```
def feature_importance_logreg(model, feature_names, df):
    w0 = model.intercept_[0]
    w = model.coef_[0]
    idx = 99
    x = df.iloc[idx][feature_names].values
    result = 0
    result += w0
    for i in range(0, 4):
        result += x[i] * w[i]
    result = sigmoid(result)
    feature_importance = pd.DataFrame(feature_names, columns=["feature"])
    feature_importance["importance"] = pow(math.e, w)
    feature_importance = feature_importance.sort_values(
        by=["importance"], ascending=False)
    ax = feature_importance.plot.barh(x='feature', y='importance')
    plt.show()
```

```
def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))
```

#### XGBClassifier

XGBClassifier has its own way to plot the feature importances:

```
pyplot.bar(range(len(bgc.feature_importances_)), bgc.feature_importances_)
pyplot.show()
```

### Validation

The validation method choosen by the competition was f1-score, which is a harmonic mean that validates the model's score based on its precision and recall, False positives,
true positives, false negatives and true negatives matrix. Implemented this way for all models:

```
f1_score(y_test, bgc.predict(x_test))
```

### Cross validation

Now we are going to compare all these models by doing a loop with random splits(default configuration of train_test_split method), so we can have a good
idea of how accurate the models are in comparison to each other. This following code is for only Random Forest, but it is analogue to other models.

```
names = ['institute_id', 'difficulty',
         'novo_user_id', 'examining_board_id',
         'gp:carrers', 'nullified',
         'outdated', 'publication_year']
# Extracted: 'country', 'device_type', 'region',
# 'scholarity_id' , 'product_id', 'knowledge_area_id',
#  'gp:segment' ,'examining_board_id'


df = create_df(names)


lista_rf = list()

for i in range(5):
    x_train, x_test, y_train, y_test = train_test_split(df.drop('acertou', axis=1), df['acertou'])
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=-1, criterion='gini', n_estimators=100, bootstrap=False)
    clf.fit(x_train.values, y_train.values)
    lista_rf.append(f1_score(y_test, clf.predict(x_test)))
#feature_importance_random_forest(clf, names)
print('------------- randomForest --------------')
print(sum(lista_rf)/len(lista_rf))
```

Running this for each model, checking every feature importance and extracting the worst features, I came to the conclusion that random forest was better fitted to this problem with those features being the ones that got me most accuracy. You may notice I also added parameters to random forest, those are included in its library and I used them so I could have the best score possible. 

### PCA dimension decomposition

PCA is Principal Component Analysis, it reduces the matrix's dimensions, its main use is for speeding up the model's calculations and data visualization,
but it can also improve your accurancy and it helped me in this model in both ways. I used scikit's PCA library and tried it on every model, but once 
again random forest responded best. So, the final model looks like this(submit.py):

```
# Modeling
names = ['institute_id', 'difficulty',
         'novo_user_id', 'examining_board_id',
         'gp:carrers', 'nullified',
         'outdated', 'publication_year']


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

y.to_csv('Submit_final.csv', index=False, header=False)
```

Turning the prediction into a csv and submitting it to the contest.

### Final conclusions

Throughout the process of developing this project i was able to learn so much about Machine Learning and acquire new programming skills. I look forward to new and even more challenging projects.

Eversince i've started working as a data science intern for a local start up, data has been my passion, what came upon wonderfully with my childhood passion, science. Bringing those two together was an amazing experience, learning along great scientist amongst the area, developing my self learning sid. It helped me learn way more than the expected, what makes me want to continue improving and colaborating with the data science comunity.


### References

> https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
> https://sefiks.com/2021/01/06/feature-importance-in-logistic-regression/#:~:text=Feature%20importance%20is%20a%20common,regression%20and%20decision%20trees%20before.
> https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
> https://xgboost.readthedocs.io/en/stable/python/python_api.html
> https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
> https://scikit-learn.org/stable/
 
> An extra special thanks to my friend and colegue Nathan whom helped me revise this document. His github: https://github.com/nac0303
 
 
 
 
 
 





# Intro to ML with TensorFlow: Finding Donors
![Mark stale issues and pull requests](https://github.com/aaronhma/finding-donors/workflows/Mark%20stale%20issues%20and%20pull%20requests/badge.svg?branch=master)
![Website Status](https://img.shields.io/badge/website-passing-brightgreen)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
<!--
![Project Status](https://img.shields.io/badge/project-in--review-orange)
![Project Status](https://img.shields.io/badge/project-meets--specification-brightgreen)
-->
![Project Status](https://img.shields.io/badge/project-requires--changes-red)

This repository contains code submitted for Udacity's Intro to ML with TensorFlow - Project 1: Finding Donors

## Contents
<!-- MarkdownTOC depth=4 -->
- [Finding Donors](https://github.com/aaronhma/finding-donors/)
  - [Getting Started](https://github.com/aaronhma/finding-donors#getting-started)
  - [Writeup](#writeup)
    - [Placeholder](#placeholder)
  - [Files](#files)
  - [Libraries Used](#libraries)
  - [Contributing](#guidelines)
  - [License](#copyright)
<!-- /MarkdownTOC -->

<a name = "setup" />

## Getting Started
1. Clone this repository.
```bash
# With HTTPS:
$ git clone https://github.com/aaronhma/finding-donors.git aaronhma-finding-donors
# Or with SSH:
$ git clone git@github.com:aaronhma/finding-donors.git aaronhma-finding-donors
```
2. Get into the repository.
```bash
$ cd aaronhma-finding-donors
```

3. Install the required dependencies.
```bash
$ pip3 install -r requirements.txt
```

4. Start the Jupyter Notebook/Jupyter Lab or click [here to access the server](http://localhost:8888).
```bash
# For jupyter notebook:
$ jupyter notebook
# Then, go to http://localhost:8888/tree?token=<YOUR TOKEN HERE>
# ============================
# For jupyter lab:
$ jupyter lab
# Then, go to http://localhost:8888/lab?token=<YOUR TOKEN HERE>
```

5. Enjoy the project!

<a name = "writeup" />

## Writeup


<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
	 width="64px" height="64px" viewBox="0 0 512 512" enable-background="new 0 0 512 512" xml:space="preserve" color="green">
<g>
	<path d="M479.971,32.18c-21.72,21.211-42.89,43-64.52,64.301c-1.05,1.23-2.26-0.16-3.09-0.85
		c-24.511-23.98-54.58-42.281-87.221-52.84c-37.6-12.16-78.449-14.07-117.029-5.59c-68.67,14.67-128.811,64.059-156.44,128.609
		c0.031,0.014,0.062,0.025,0.093,0.039c-2.3,4.537-3.605,9.666-3.605,15.1c0,18.475,14.977,33.451,33.451,33.451
		c15.831,0,29.084-11.002,32.555-25.773c19.757-41.979,58.832-74.445,103.967-85.527c52.2-13.17,111.37,1.33,149.4,40.041
		c-22.03,21.83-44.391,43.34-66.33,65.26c59.52-0.32,119.06-0.141,178.59-0.09C480.291,149.611,479.931,90.891,479.971,32.18z"/>
	<path d="M431.609,297.5c-14.62,0-27.041,9.383-31.591,22.453c-0.009-0.004-0.019-0.008-0.027-0.012
		c-19.11,42.59-57.57,76.219-102.84,88.18c-52.799,14.311-113.45,0.299-152.179-39.051c21.92-21.76,44.369-43.01,66.189-64.869
		c-59.7,0.049-119.41,0.029-179.11,0.01c-0.14,58.6-0.159,117.189,0.011,175.789c21.92-21.91,43.75-43.91,65.79-65.699
		c14.109,13.789,29.76,26.07,46.92,35.869c54.739,31.971,123.399,38.602,183.299,17.891
		c57.477-19.297,106.073-63.178,131.212-118.318c3.645-5.357,5.776-11.824,5.776-18.793C465.06,312.477,450.083,297.5,431.609,297.5
		z"/>
</g>
</svg> Successfully synced with project notebook!


### 0. Preparation Work
#### Installing Packages
Before we begin the project, we must install all the packages. Follow along to get your own copy of the packages used in this project.

```bash
# If you're in the Terminal:
$ pip install -r requirements.txt
# Or if you're in Jupyter:
!pip install -r requirements.txt
```

#### Loading Packages
After installing the packages we must load it for the project.

```python
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs
```

### 1. Exploring the Data
#### Install the data
In order to install the data, please sign in to your classroom to install it.

#### Load the data
Assuming you've installed the data, we'll load the data with [Pandas](pandas.pydata.org).

```python
# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))
```

#### Data Exploration
Now, it's time to actually start the project!

```python
# Total number of records
n_records = len(data)

# Number of records where individual's income is more than $50,000
n_greater_50k = len(data[data.income == '>50K'])

# Number of records where individual's income is at most $50,000
n_at_most_50k = len(data[data.income == '<=50K'])

# Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k / n_records) * 100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
```

#### Preparing the Data

```python
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)
```

```python
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)
```

```python
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = records_shown))
```

#### Data Preprocessing

```python
# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
income = income_raw.map({"<=50K": 0, ">50K": 1})

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
display(encoded)
```

#### Shuffle and Split the Data

```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
```

#### Evaluating Model Performance

```python
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
# encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

# Calculate accuracy, precision and recall
accuracy = (TP + TN) / (TP + FP + TN + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta = 0.5
beta_sq = beta * beta
fscore = (1 + beta_sq) * ((precision * recall) / ((beta_sq * precision) + recall))

# Print the results
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
```

#### Creating a Training & Predicting Pipeline

```python
# Import metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test, beta = 0.5): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:300], y_train[:300])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_train[:300])
    predictions_train = learner.predict(X_test[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test[:300], predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_train[:300], predictions_train, beta)
       
    # Success
    print("{} trained on {} samples with {} accuracy on train and {} accuracy on test.".format(learner.__class__.__name__, sample_size, results['acc_train'], results['acc_test']))
        
    # Return the results
    return results
```

#### Initial Model Evaluation

```python
# Import the three supervised learning models from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize the three models
clf_A = RandomForestClassifier(random_state = 1)
clf_B = AdaBoostClassifier(random_state = 1)
clf_C = DecisionTreeClassifier(random_state = 1)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(y_train)
samples_10 = int(samples_100 * 0.1)
samples_1 = int(samples_100 * 0.01)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
```

#### Model Tuning

```python
# Import 'GridSearchCV', 'make_scorer'
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Initialize the classifier
clf = AdaBoostClassifier(random_state = 1)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 50, 75, 100],
    'algorithm': ['SAMME', 'SAMME.R'],
    'learning_rate': [0.01, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta = beta)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = beta)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = beta)))
```

#### Extracting Feature Importance

```python
# Train the supervised model on the training set using .fit(X_train, y_train)
model = AdaBoostClassifier(random_state = 1)
model.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)
```

#### Feature Selection

```python
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
```

<a name = "files" />

## Files

Here's the current, up-to-date, file structure for this project:

```
|____CODE_OF_CONDUCT.md
|____LICENSE
|____requirements.txt
|____finding_donors.ipynb
|____README.md
|____.gitignore
|____.github
| |____FUNDING.yml
| |____workflows
| | |____stale.yml
| | |____greetings.yml
|____visuals.py
|____SECURITY.md
```

<a name = "libraries" />

## Libraries Used

The libraries used for this project can be found [here](https://github.com/aaronhma/finding-donors/blob/master/requirements.txt).

<a name = "guidelines" />

## Contributing
Contributions are always welcome!

<a name = "copyright" />

## License
The MIT License is used for this project, which you can read [here](https://github.com/aaronhma/aitnd-momentum-trading/blob/master/LICENSE):

```
MIT License

Copyright (c) 2020 - Present Aaron Ma,
Copyright (c) 2020 - Present Udacity, Inc.
All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

In short, here's the license:
```
A short and simple permissive license with conditions only requiring
preservation of copyright and license notices.
Licensed works, modifications, and larger works may be
distributed under different terms and without source code.
```

| Permissions                      | Allowed?           |
| -------                          | ------------------ |
| Commercial use                   | :white_check_mark: |
| Modification                     | :white_check_mark: |
| Re-distribution (with license)   | :white_check_mark: |
| Private use (with license)       | :white_check_mark: |
| Liability                        | :x:                |
| Warranty                         | :x:                |

**The checkmarked items are allowed with the condition of the original license and copyright notice.**

For more information, click [here](https://www.copyright.gov/title17/title17.pdf).

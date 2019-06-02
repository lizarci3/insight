#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:55:06 2019

@author: liz
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
import time
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

start_time = time.time()

'''
This script is used to model and predict the PRICES of 
active Airbnb listings in the city of Toronto in
May 2019
'''

def PolynomialRegression(degree, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))

def plot_histogram_features(features_to_plot, dataframe):
    
    to_plot = dataframe[features_to_plot]
    to_plot.hist(figsize=(20,15))
    plt.show()

def correlation_matrix(features_to_plot, listings):
    
    to_plot = listings[features_to_plot]
    corr_matrix = to_plot.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr_matrix, square=True, vmin=-1, vmax=1)

def plot_hists_price_occupancy(y_price, lower_limit, upper_limit):
    
    
    plt.figure()
    plt.title('Prices, Active listings Toronto 2019')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.hist(y_price,  bins=30, color='cyan', range=[lower_limit, upper_limit])

def clean_not_available_info(listings):
    '''
    Make sure you have the information from the features
    you will use for your fit
    '''
    
    listings = listings[np.isfinite(listings['bathrooms'])]
    listings = listings[np.isfinite(listings['bedrooms'])]
    listings = listings[np.isfinite(listings['beds'])]
    
    return(listings)
    
def map_active_listings(listings, lower_range, upper_range):
    
    listings['normalized_price'] = listings['price_clean']/listings['accommodates']
    listings.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5, title='Active Listings Airbnb, Toronto 2019')
    listings.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, c='normalized_price', cmap=plt.get_cmap("jet"), colorbar=True)
    
    
    
    plt.figure()
    plt.title('Price per person distribution, Toronto 2019')
    plt.xlabel('Price/person')
    plt.ylabel('Frequency')
    plt.hist(listings['price_clean']/listings['accommodates'], bins=30, range=[lower_range,upper_range], color='green')
    
def training_validation_performance(y_price, work_with_label, fit, yval, ytrain, Xval, Xtrain):
    
    x_prices = np.arange(0,max(y_price),1)
    plt.figure()
    plt.title('Validation Set Price Prediction '+work_with_label)
    plt.xlabel('Validation Real Prices')
    plt.ylabel('Validation Predicted Prices')
    plt.scatter(yval, fit.predict(Xval), color='blue')
    plt.plot(x_prices, x_prices, linestyle='--', color='black')
    plt.show()
    
    plt.figure()
    plt.title('Training Set Price Prediction '+work_with_label)
    plt.xlabel('Train Real Prices')
    plt.ylabel('Train Predicted Prices')
    plt.scatter(ytrain, fit.predict(Xtrain), color='green')
    plt.plot(x_prices, x_prices, linestyle='--', color='black')
    plt.show()
    
    
##############################################################################################

listings = pd.read_csv('/home/liz/Desktop/Airbnb/week3/listings_dummies_v3.csv', error_bad_lines = False)


### Some rows don't have the number of bathrooms nor bedrooms
listings = clean_not_available_info(listings)

### plot a distribution of your features
to_hist = ['bathrooms','bedrooms','accommodates', 'av_occupancy_may', 'distance_from_center', 'latitude', 'longitude', 'review_scores_rating']
plot_histogram_features(to_hist, listings)
sns.pairplot(listings[['price_clean','bathrooms', 'bedrooms', 'accommodates', 'distance_from_center']], size=1.5)

### Map of active listings
map_active_listings(listings, 0, 600)


###############################################################################################
### Feature engineering
###############################################################################################

### This correlation matrix misses any non-linear correlations
corr_matrix = listings.corr()
print(corr_matrix['price_clean'].sort_values(ascending=False))


for_corr = ['bathrooms','bedrooms','distance_from_center', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'accommodates', 'price_clean']
correlation_matrix(for_corr, listings)


##############################################################################################
### Modeling
##############################################################################################

### Define wich features you will be working with
features = ['bathrooms', 'bedrooms', 'distance_from_center', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'accommodates']
X = listings[features]

X = X.values # when saving your model to pickle from xgboost to flask you need to work with .values

## Establish your label (target value, i.e., price)
y_price = (listings['price_clean'])


'''
Using y_price and y_occupancy.describe() I could check my outliers
and verify that they are real (i.e., zero price or really expensive)
'''

plot_hists_price_occupancy(y_price, 0, 1000)


## Obtain your Training, validation and test samples
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_price, test_size=0.3, random_state=7)
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.1, random_state=7)


###############################################################################################
## Define the models you will be working with
## Since my initial models started underfitting I increased
## the complexity of the model from linear regression to decision
## and random trees and finally found the best performance with XGBoost
##############################################################################################



## Establish your starting models
lin_reg = LinearRegression()
dec_tree = DecisionTreeRegressor()
ran_tree = RandomForestRegressor()
xg = xgb.XGBRegressor(learning_rate = 0.1, colsample_bytree = 0.3, max_depth=5, alpha=10) 

## To change between models simply assign a different model to work_with
work_with = xg
work_with_label = 'XGBoost Regressor'#'Random Forest Regressor'


#### Fit check your training and validation set outputs 

fit = work_with.fit(Xtrain, ytrain)
joblib.dump(fit, "pricing_model.pkl")

### See the performance of your training and validation sets
training_validation_performance(y_price, work_with_label, fit, yval, ytrain, Xval, Xtrain)


############# Compare with cross-validation

## Train your model
score_to_use = ['neg_mean_squared_error', 'neg_median_absolute_error', 'r2']

for score in score_to_use:
    print('Working with ', score)
    scores_cv = cross_val_score(work_with, Xtrain, ytrain, cv=10, scoring=score)
    
    if(score=='neg_mean_squared_error'):
        print('RMSE: {:.2f}'.format(np.sqrt(abs(scores_cv.mean()))))
        
    elif(score=='neg_median_absolute_error'):
        print('Median Absolute error: {:.2f}'.format(abs(scores_cv.mean())))
    
    elif(score=='r2'):
        print('R2: {:.2f}'.format(scores_cv.mean()))



print ("My program took", (time.time() - start_time)/60, " min to run")

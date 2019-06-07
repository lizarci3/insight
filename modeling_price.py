#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:15:59 2019

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
from sklearn.model_selection import GridSearchCV, cross_val_score, LeaveOneOut, train_test_split
import time
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib
from matplotlib.pyplot import figure
from mpl_toolkits.basemap import Basemap
import mplleaflet

plt.rcParams["figure.figsize"] = (8,8)

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
    
    #sns.set(font_scale=3)
    f, ax = plt.subplots(figsize=(14, 14))
    
    ax = sns.heatmap(corr_matrix, square=True, vmin=-1, vmax=1, cmap='Blues')
    #ax.figure.axes[-1].yaxis.label.set_size(20)
    #cbar.ax.tick_params(labelsize=10) 
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    ax.tick_params(labelsize=17)
    plt.tight_layout()
    plt.savefig('test_price.png')
    
def plot_hists_price_occupancy(y_price, lower_limit, upper_limit):
    
    
    plt.figure(figsize=(8,8))
    plt.title('Prices, Active Listings Toronto 2019', fontsize=19)
    plt.xlabel(r'Log$_{10}$(Price)', fontsize=17)
    plt.ylabel('Number of Listings', fontsize=17)
    #plt.xlim(0,1100)
    #plt.ylim(0,4500)
    plt.xlim(1, 3.5)
    plt.ylim(0, 2500)
    plt.xticks(fontsize=15)
    #plt.yticks([1000, 2000, 3000, 4000], fontsize=15)
    plt.yticks([0, 500, 1000, 1500, 2000], fontsize=15)
    plt.hist(y_price,  bins=30, color='#4681AA', range=[lower_limit, upper_limit])
    #plt.hlines([1000, 2000, 3000, 4000], 0, 6000, colors='black', linestyles=':')
    plt.hlines([0, 500, 1000, 1500, 2000], 0, 4, colors='black', linestyles=':')
    '''
    for_sns = y_price[y_price<upper_limit]
    plt.figure(figsize=(8,8))
    sns.distplot(for_sns, bins=20, hist=True, kde=False, norm_hist=False)
    plt.xlabel('Price', fontsize=17)
    plt.ylabel('Frequency', fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks([1000, 2000, 3000, 4000], fontsize=15)
    '''

def clean_not_available_info(listings):
    '''
    Make sure you have the information from the features
    you will use for your fit
    '''
    
    listings = listings[np.isfinite(listings['bathrooms'])]
    listings = listings[np.isfinite(listings['bedrooms'])]
    listings = listings[np.isfinite(listings['beds'])]
    
    return(listings)

def map_active_listings2(listings):
    
    plt.figure()
    plt.scatter(listings['longitude'], listings['latitude'], alpha=0.3)
    mplleaflet.show()
    
def map_active_listings(listings, lower_range, upper_range):
    
    
    listings_map = listings[listings['price_clean']< upper_range]
    listings_map['normalized_price'] = listings_map['price_clean']/listings_map['accommodates']
    
    
    ax = listings_map.plot(kind='scatter', x='longitude', y='latitude', alpha=0.5, title='Active Listings Airbnb, Toronto 2019')
    #ax = listings_map.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, c='normalized_price', cmap=plt.get_cmap("jet"), colorbar=True, title='Active Listings Airbnb, Toronto 2019')
    ax.set_xlabel('Longitude', fontsize=17)
    ax.set_ylabel('Latitude', fontsize=17)
    ax.set_xticks([-79.6, -79.4, -79.2])
    ax.set_xticklabels([-79.6, -79.4, -79.2], fontsize=15)
    
    ax.set_yticks([43.60, 43.70, 43.80])
    ax.set_yticklabels([43.60, 43.70, 43.80], fontsize=15)
    
    ax.set_title('Active Listings Airbnb, Toronto 2019', fontsize=20)
    print('Min price ', min(listings_map['price_clean']), 'Max price ', max(listings_map['price_clean']))
    
    
    #plt.figure()
    #plt.title('Price per person distribution, Toronto 2019')
    #plt.xlabel('Price/person')
    #plt.ylabel('Frequency')
    #plt.hist(listings['price_clean']/listings['accommodates'], bins=30, range=[lower_range,upper_range], color='green')
    
def training_validation_performance(y_price, work_with_label, fit, yval, ytrain, Xval, Xtrain):
    
    x_prices = np.arange(0,max(y_price)+1,1)
    plt.figure()
    plt.title('Validation Set Price Prediction '+work_with_label)
    plt.xlabel('Validation Real Prices')
    plt.ylabel('Validation Predicted Prices')
    plt.xlim(1, 3.5)
    plt.ylim(1, 3.5)
    plt.scatter(yval, fit.predict(Xval), color='blue')
    plt.plot(x_prices, x_prices, linestyle='--', color='black')
    plt.show()
    
    plt.figure()
    plt.title('Training Set Price Prediction '+work_with_label)
    plt.xlabel('Train Real Prices')
    plt.ylabel('Train Predicted Prices')
    plt.xlim(1, 3.5)
    plt.ylim(1, 3.5)
    plt.scatter(ytrain, fit.predict(Xtrain), color='green')
    plt.plot(x_prices, x_prices, linestyle='--', color='black')
    plt.show()
    
    
##############################################################################################

listings = pd.read_csv('/home/liz/Desktop/Airbnb/week4/listings_dummies_v3.csv', error_bad_lines = False)


### Some rows don't have the number of bathrooms nor bedrooms
listings = clean_not_available_info(listings)

### plot a distribution of your features
#to_hist = ['bathrooms','bedrooms','accommodates', 'av_occupancy_may', 'distance_from_center', 'latitude', 'longitude', 'review_scores_rating']
#plot_histogram_features(to_hist, listings)
#sns.pairplot(listings[['price_clean','bathrooms', 'bedrooms', 'accommodates', 'distance_from_center']], size=1.5)

### Map of active listings
#map_active_listings(listings, 0, 1000)
#map_active_listings2(listings)

###############################################################################################
### Feature engineering
###############################################################################################

### This correlation matrix misses any non-linear correlations
#corr_matrix = listings.corr()
#print(corr_matrix['price_clean'].sort_values(ascending=False))


#for_corr = ['bathrooms','bedrooms','distance_from_center', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'accommodates', 'price_clean']
#correlation_matrix(for_corr, listings)


##############################################################################################
### Modeling
##############################################################################################

### Define wich features you will be working with
features = ['bathrooms', 'bedrooms', 'distance_from_center', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'accommodates']
X = listings[features]

X = X.values # when saving your model to pickle from xgboost to flask you need to work with .values

## Establish your label (target value). To obtain a normal
## distribution of prices we use the Log10 of price

y_price = np.log10(listings['price_clean'])
#y_price = (listings['price_clean'])


'''
#Using y_price and y_occupancy.describe() I could check my outliers
#and verify that they are real (i.e., zero price or really expensive)
'''

plot_hists_price_occupancy(y_price, 1, 3.5)
#plot_hists_price_occupancy(y_price, 0, 1000)


'''
## Obtain your Training, validation and test samples
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_price, test_size=0.3, random_state=7)
Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, test_size=0.1, random_state=7)


###############################################################################################
## Define the models you will be working with
## Since my initial models started underfitting I increased
## the complexity of the model from linear regression to decision
## and random trees and finally found the best performance with XGBoost
##############################################################################################

param_grid = { 
    'learning_rate':[0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'max_depth' : [4,5,6,7],
    'min_samples_leaf' : [5, 10],
    'colsample_bytree': [0.3, 0.5]
}

## Establish your starting models
lin_reg = LinearRegression() #underfits
dec_tree = DecisionTreeRegressor() #underfits
ran_tree = RandomForestRegressor() #OK but not great, increase model complexity
#xg = xgb.XGBRegressor()#(learning_rate = 0.1, colsample_bytree = 0.3, max_depth=5, alpha=10) 
xg = xgb.XGBRegressor(learning_rate = 0.1, colsample_bytree = 0.5, max_depth=6, n_estimators=100, min_samples_leaf=5) 


work_with = xg
work_with_label = 'XGBoost Regressor'#'Random Forest Regressor'

## To change between models simply assign a different model to work_with
## Using CV Grid search estimate hyperparameter tuning
## From this output I decided on the tuning you see above for xgb
'''

'''
grid_search = GridSearchCV(estimator=work_with, param_grid=param_grid, cv=5, scoring='r2')
fit_model = grid_search.fit(Xtrain, ytrain)
print(fit_model.best_params_)
final_model = grid_search.best_estimator_
'''

'''
#### Fit check your training and validation set outputs 

fit = final_model.fit(Xtrain, ytrain)
joblib.dump(fit, "logpricing_model_hyperparam.pkl")

### See the performance of your training and validation sets
training_validation_performance(y_price, work_with_label, fit, yval, ytrain, Xval, Xtrain)


############# Compare the output of your models with different scoring metrics

test_prediction = final_model.predict(Xtest)
r2_score = r2_score(ytest, test_prediction)
mse_score = mean_squared_error(ytest, test_prediction)
rmse_score = np.sqrt(mse_score)
medabse = median_absolute_error(ytest, test_prediction)

print('R2', r2_score)
print('RMSE ', rmse_score)
print('Median absolute error ', medabse)


print ("My program took", (time.time() - start_time)/60, " min to run")
'''
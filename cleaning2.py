#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:37:34 2019

@author: liz
"""

import pandas as pd
import numpy as np

def clean_prices(listings):
    
    '''
    #Prices started with a $ sign and thousands were represented with ","
    #so we need to clean this as well
    '''
    
    listings['price_clean'] = np.nan
    listings['price_clean'] = [float(p[1:].replace(",", "")) for p in list(listings['price'])]
    new_listings = listings.drop(columns=['price'], axis=1)
    
    
    return(new_listings)

    
listings = pd.read_csv('/home/liz/Desktop/Airbnb/week3/clean_part1_v3.csv', error_bad_lines = False, index_col=0)
#listings = listings[100:500] #work with a subset of your data, for testing purposes only
listings_priceclean = clean_prices(listings)


'''
#We want to transform our categorical features into nulistings = listings.loc[:, ~listings.columns.str.contains('^Unnamed')]mbers
#using LabelEncoders
'''


'''
#Get the dummy variables
'''

listings_dummies = pd.get_dummies(listings_priceclean, columns=['room_type'])

## Save to file
#listings_dummies.to_csv('/home/liz/Desktop/Airbnb/week3/listings_dummies_v3.csv', sep=',')


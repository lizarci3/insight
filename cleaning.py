#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:50:27 2019

@author: liz
"""

'''
Cleaning the data part 1 for part two run afterwards
cleaning2.py
'''
import pandas as pd
import numpy as np
from geopy import distance, Point

def clean_shifted(amenities, initial_array):
    
    '''
    Clean the format of output amenties, leaving them as floats,
    without nan values and removing any strings
    
    Clean any rows where the information was not stored properly (it seems
    during scraping everything shifted for a few rows)
    
    Eventually I realized that the parsing the website uses to separate
    fields (,) was mistaken in some fields, so for one field 
    where they have the description of their homes, if a comma
    was used in the description, this comma at times separated fields
    '''
    
    array = initial_array
    
    for am in amenities:
        array = array.dropna(axis=0, subset=[am])
        array = array[pd.to_numeric(array[am], errors='coerce').notnull()]
        array[am] = array[am].astype(float)
    
    
    array = array[((array['has_wifi']==1) | (array['has_wifi']==0))]
    array = array[((array['has_kitchen']==1) | (array['has_kitchen']==0))]
    array = array[((array['has_free_parking']==1) | (array['has_free_parking']==0))]
    

    
    return(array)
    
def remove_idle_users(min_days, recent_listings_woccupancy):
    
    '''
    #First remove hosts that are relatively new (for now I ask that they are
    #at least 1 month old in the platform, limit set by min_days)
    '''

   
    minimum_days = pd.to_timedelta(min_days, unit='day')
    listings_wocc_thost = recent_listings_woccupancy[recent_listings_woccupancy['host_since_time']>minimum_days]  
    
    '''
    #Sometimes it could be old hosts with new listings, still
    #remove listings with no reviews (assumes no reviews=new listing improve after MVP)
    '''
    listings_wocc_thost['number_of_reviews'] = listings_wocc_thost['number_of_reviews'].astype(int)
    listings_wocc_thost = listings_wocc_thost[listings_wocc_thost['number_of_reviews']>0]

    return(listings_wocc_thost)

def calculate_distance_from_center(listings_wocc_thost):
    
    downtown_lat = 43.6426
    downtown_lon = -79.3871
    
    center = Point(downtown_lat, downtown_lon)
    distance_vec = []
    
    listings_wocc_thost['latitude'] = listings_wocc_thost['latitude'].astype(float)
    listings_wocc_thost['longitude'] = listings_wocc_thost['longitude'].astype(float)
    
    for index_point, row_point in listings_wocc_thost.iterrows():
        p = Point(row_point['latitude'], row_point['longitude'])
        d_center = distance.distance(p, center).kilometers
        distance_vec.append(d_center)
        
    return(distance_vec)
    

now = pd.to_datetime('2019-05-01')
listings = pd.read_csv('/home/liz/Desktop/Airbnb/week3/listings_occ_amenities.csv', error_bad_lines = False, index_col=0)
print('INitial shape ', listings.shape)

listings = listings.loc[:, ~listings.columns.str.contains('^Unnamed')]
testings_start = listings#[13000:13500] # if you want to work with a subset of your data

############################################################################################
### Remove any lines where the parsing didn't work as well (i.e., all fields are shifted)
############################################################################################
amenities = ['has_wifi', 'has_kitchen', 'has_free_parking']

listings_no_shift = clean_shifted(amenities, testings_start)
listings_no_shift = listings_no_shift.dropna(axis=0, subset=['host_since'])

flag_wrong_info = listings_no_shift['host_since'].str.startswith('20')
recent_listings_woccupancy = listings_no_shift[flag_wrong_info]

print('Shape after flagging ', recent_listings_woccupancy.shape)


############################################################################################
### Remove idle users by setting a min amount of days in the platform
### also removes old hosts new listings
#############################################################################################

recent_listings_woccupancy['host_since_date']= pd.to_datetime(recent_listings_woccupancy['host_since'])
recent_listings_woccupancy['host_since_time']= now-recent_listings_woccupancy['host_since_date']


listings = listings.loc[:, ~listings.columns.str.contains('^Unnamed')]

min_days = 30
listings_wocc_thost = remove_idle_users(min_days, recent_listings_woccupancy)


### Calculate a new feature: Distance from CN Tower
distance_vec = calculate_distance_from_center(listings_wocc_thost)    
listings_wocc_thost['distance_from_center'] = distance_vec

cols_to_save = ['id', 'latitude', 'longitude', 
                'distance_from_center', 'host_since', 
                'host_response_rate', 'host_is_superhost', 
                'neighbourhood', 'property_type', 
                'room_type', 'accommodates', 'amenities',
                'has_kitchen', 'has_wifi', 
                'has_free_parking', 'price', 
                'review_scores_rating', 'av_occupancy_may', 'property_type',
                'bathrooms', 'bedrooms', 'beds']


## Save to file
#listings_wocc_thost.to_csv('/home/liz/Desktop/Airbnb/week3/clean_part1_v2_allcols_widle.csv', sep=',', index=False)
#listings_wocc_thost.to_csv('/home/liz/Desktop/Airbnb/week3/clean_part1_v3.csv', sep=',', index=False, columns=cols_to_save)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:50:18 2019

@author: liz
"""

import pandas as pd
import numpy as np
import re
import time


start_time = time.time()

recent_listings_woccupancy = pd.read_csv('/home/liz/Desktop/Airbnb/week3/listings_occupancy_avmay.csv', error_bad_lines = False)
testings_start = recent_listings_woccupancy#[200:207] #work with a subsection of your data


'''
Clean any rows that the information was not stored properly (it seems
during scraping everything shifted for a few rows)
'''
testings_start = testings_start.dropna(axis=0, subset=['host_since'])
testings_start = testings_start.dropna(axis=0, subset=['amenities'])
flag_wrong_info = testings_start['host_since'].str.startswith('20')
testings = testings_start[flag_wrong_info]



wifi = []
kitchen = []
parking = []

for listing_index, listing_row in testings.iterrows():
    print('ID ', listing_row['id'])
    
    has_wifi = 0
    has_kitchen = 0
    has_free_parking = 0
    
    
    amenities = listing_row['amenities']
    
    #Remove unwanted characters from the field
    am_clean = amenities.replace('{','').replace('}','').replace('"','')
    
    # Split the list to ID individual amenities
    am_sep = am_clean.split(',')
    am_sep_lower = [z.lower() for z in am_sep]
    
    
    has_wifi = 1 if 'wifi' in am_sep_lower else 0
    has_kitchen = 1 if 'kitchen' in am_sep_lower else 0
   
    ## Since free parking can be written using different
    ## expressions, use regular expression search for
    ## this flag
    
    for each_am in am_sep_lower:

        
        if (re.search('free parking', each_am)) != None:
            has_free_parking = 1
            break
        
    wifi.append(has_wifi)
    kitchen.append(has_kitchen)
    parking.append(has_free_parking)
    
 
testings['has_wifi'] = wifi
testings['has_kitchen'] = kitchen
testings['has_free_parking'] = parking

# Save to file
#testings.to_csv('/home/liz/Desktop/Airbnb/week3/testings_amenities.csv', index=False)  

print('It took ', time.time() - start_time, 'seconds')

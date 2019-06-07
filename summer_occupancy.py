#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:12:19 2019

@author: liz
"""


'''
Extract the occupancy information for the summer months
'''

import pandas as pd
import numpy as np
import os
import time


'''
FInd the information of summer occupancy for 
previous years to determine this year's estimated
occupancy
'''

start_time = time.time()

all_years = np.arange(2016, 2019, 1)
months = ['07', '08']

recent_listings = pd.read_csv('/home/liz/Desktop/Airbnb/data_downloading/2019-05_listings.csv', error_bad_lines = False, low_memory=False)
#recent_listings = recent_listings.iloc[0:10]

'''
Add a column that shows me occupancy for any given listing
in days for the months of summer
'''

for year in all_years:
    print('working with year ', year)
    
    year = str(year)
    
    for month in months:
        
        recent_listings['occupancy_'+month+'_'+year] = np.nan
        start_vec = []
        
        file_calendar = '/home/liz/Desktop/Airbnb/data_downloading/'+year+'-'+month+'_calendar.csv'
        
        
        if os.path.isfile(file_calendar) == True:
            #print('I exist!')
            
            calendar = pd.read_csv(file_calendar)
        
            
            for index_listing, row_listing in recent_listings.iterrows():
                
                
                listing_id = float(row_listing['id'])
                
                listing_in_calendar = calendar[calendar['listing_id']==listing_id]
                
                days_occupied = listing_in_calendar[listing_in_calendar['available']=='f']
                
                
                '''
                We only care about days occupied at that given time in the calendar
                (i.e., we determine May occupancy using the calendar for May), 
                the forecasting in these catalogs can be highly incomplete
                hence it is not used
                '''
                days_in_month = 0
                
                for index_days_occ, row_days_occ in days_occupied.iterrows():
                    
                    
                    working_year = year+'-'+month
                    if (row_days_occ['date'][0:7]==working_year):
                        days_in_month = days_in_month + 1
                        #print('Days in month', month, days_in_month)
                
                
                start_vec.append(days_in_month)
                #print('Days occupied Final ', month, year, days_in_month)
                
                #recent_listings['occupancy_'+month+'_'+year].iloc[index_listing] = days_in_month
                       
        recent_listings['occupancy_'+month+'_'+year] = start_vec
                
        
recent_listings.to_csv('test_calendar.csv', sep=',')
print ("My program took", (time.time() - start_time)/60, " min to run")        


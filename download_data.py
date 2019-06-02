#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:42:58 2019

@author: liz
"""

'''
See if you can streamline the download of the data from
airbnb
'''
import urllib.request
import numpy as np
import requests
import time
import os.path


'''
The name the file was saved under includes the date when
the data was scraped, he normally did this the first few days of the month
but is random, so I will check all first 15 days of the month
to see if the file exist

Some months are missing completely
'''

start_time = time.time()


years = np.arange(2015, 2020, 1) 
months = np.arange(1,13,1)
days = np.arange(1,16,1)
kinds_of_files = ['listings', 'calendar', 'reviews']


for kind in kinds_of_files:
    
    print('Downloading file ', kind)
    
    for year in years:
        print('Year', year)
        
        for month in months:
            
            if month < 10:
                month = '0'+str(month)
            else:
                month = str(month)
                
            print('Month', month)
            
            for day in days:
                
                if day < 10:
                    day = '0'+str(day)
                else:
                    day = str(day)
                    
                print('Day', day)    
                
                url = 'http://data.insideairbnb.com/canada/on/toronto/'+str(year)+'-'+month+'-'+day+'/data/'+kind+'.csv.gz'
                file_name = '/home/liz/Desktop/Airbnb/data_downloading/'+str(year)+'-'+month+'_'+kind+'.csv.gz'
                
                
                if os.path.isfile(file_name) == False:
                
                    request = requests.get(url)
                    
                    if request.status_code == 200:
                        
                        urllib.request.urlretrieve(url, file_name)
                        
                    else:
                        print('URL does not exist')
                                    
                    
                    ## To not overwhelm the website with requests, put a time delay
                    time.sleep(1)
                
                else:
                    
                    print('File exists')
                    break
                    
print ("My program took", time.time() - start_time, "to run")
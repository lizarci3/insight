#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:21:08 2019

@author: liz
"""

import pandas as pd
import csv

def print_information_data(data_series, data):
    
    print('Printing Describe')
    print(data_series.describe())
    
    print('Printing Head')
    print(data_series.head())
    
    print('Printing Info')
    print(data.info())
    


def create_csvfile(file_name):
    
    '''
    Opens a csv file for writing in a Latin-1 encoding and creates
    a csv writer instance with the excel dialect.
    Remember to close file at the end of your script
    '''
    
    file = open(file_name, 'w', newline='', encoding='Latin-1')
    writer_file = csv.writer(file, dialect='excel')
    
    return(file, writer_file)
    
def add_zero_in_front(number):
    if number < 10:
        str_number = '0'+str(number)
    else:
        str_number = str(number)
        
    return(str_number)
                

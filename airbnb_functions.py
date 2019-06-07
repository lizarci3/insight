from sklearn.externals import joblib
from geopy import distance, Point
import numpy as np
import pandas as pd
import numpy as np
from numpy.random import seed, randint
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from geopy.geocoders import Nominatim


def load_model(file_name):
	model = joblib.load(file_name)
	return (model)

def calculate_distance_from_center(latitude, longitude):
    
    downtown_lat = 43.6426
    downtown_lon = -79.3871
    
    center = Point(downtown_lat, downtown_lon)
    p = Point(float(latitude), float(longitude))
    d_center = distance.distance(p, center).kilometers
 
   
    return(d_center)
    
def predict_price(model, latitude, longitude, entire_home, private_room, shared_room, has_wifi, has_kitchen, has_free_parking, accommodates, bathrooms, bedrooms, distance_center):
    vec = [bathrooms, bedrooms, distance_center, entire_home, private_room, shared_room, has_wifi, has_kitchen, has_free_parking, accommodates]
    vec_2d = np.array(vec).reshape((1,-1))
    distance_center = calculate_distance_from_center(latitude, longitude)
    price_predicted = model.predict(vec_2d)
    return(price_predicted)



def remove_newer_hosts(listings):
    min_date = '01-01-2016'
    listings['host_since_date']= pd.to_datetime(listings['host_since'])
    
    date_dt3 = datetime.strptime(min_date, '%m-%d-%Y')
    listings = listings[listings['host_since_date'] < date_dt3]
    listings = listings[listings['av_occupancy_may']>0] 
    
    return(listings)


##############################################################################################################

def descriptive_statistics(new):
    
    listings = pd.read_csv('listings_summer_calendar.csv', error_bad_lines = False, usecols=['bathrooms', 'bedrooms', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'price_clean', 'av_occupancy_may', 'av_occupancy_june', 'av_occupancy_july', 'av_occupancy_august', 'review_scores_rating', 'host_since'])
    
    listings = listings.dropna(axis=0)
    
    ## Keep only older hosts (host since 2016) to obtain good occupancy statistics
    
    listings = remove_newer_hosts(listings)
    
    
    ## Select on what features you will calculate similarity
    for_similarity = ['bathrooms', 'bedrooms', 'room_type_Entire home/apt', 'room_type_Private room', 'room_type_Shared room', 'has_wifi', 'has_kitchen', 'has_free_parking', 'price_clean']
    to_compare = listings[for_similarity]
    
    new = np.array(new)
    new = new.reshape(1, -1)
    
    
    all_sims = []
    
    
    for each_index, each_row in to_compare.iterrows():
        
        each_row = np.array(each_row)
        each_row = each_row.reshape(1, -1)
        temp_sim = cosine_similarity(each_row, new)
        
        all_sims.append(temp_sim[0][0])
     
    listings['similarity'] = np.nan
    listings['similarity'] = all_sims
    
    sorted_similarities = listings.sort_values(by='similarity', ascending=False)
    
    ## Choose the top 5% most similar listings to this new listing
    top_5p_number = int(5*len(sorted_similarities)/100)
    top_5p = sorted_similarities.iloc[0:top_5p_number]
    
    av_may = round(np.average(top_5p['av_occupancy_may']),1)
    av_june = round(np.average(top_5p['av_occupancy_june']),1)
    av_july = round(np.average(top_5p['av_occupancy_july']),1)
    av_august = round(np.average(top_5p['av_occupancy_august']),1)
    av_review = round(np.average(top_5p['review_scores_rating']),1)
    
    return(av_may, av_june, av_july, av_august, av_review)
    
    
def get_latlon_from_address(address):
    
    geolocator = Nominatim()
    city = "Toronto"
    country = "CA"
    
    location = geolocator.geocode(address+' '+city+' '+country)
    return(location.latitude, location.longitude)
    
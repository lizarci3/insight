from flask import Flask, render_template, request
import airbnb_functions as airbnb
import numpy as np

# Create the application object
app = Flask(__name__)


@app.route('/',methods=["GET","POST"])
def home_page():
    return render_template('index_v2.html')  # render a template

@app.route('/output',methods=["POST"])
def tag_output():
      
       # Pull information from input field 
       model_pricing = airbnb.load_model('logpricing_model_hyperparam.pkl')

       address = request.form.get('address')
       #latitude = request.form.get('latitude')
       #longitude = request.form.get('longitude')
       entire_home = request.form.get('room_type_Entire home/apt')
       
       if entire_home is None:
           entire_home = 0
           
       private_room = request.form.get('room_type_Private room')
       
       if private_room is None:
           private_room = 0
           
       shared_room = request.form.get('room_type_Shared room')
       
       if shared_room is None:
           shared_room = 0
           
       has_wifi = request.form.get('has_wifi')
       
       if has_wifi is None:
           has_wifi = 0
           
       has_kitchen = request.form.get('has_kitchen')
       
       if has_kitchen is None:
           has_kitchen= 0
           
           
       has_free_parking = request.form.get('has_free_parking')
       
       if has_free_parking is None:
           has_free_parking = 0
           
       accommodates = request.form.get('accommodates')
       bedrooms = request.form.get('bedrooms')
       bathrooms = request.form.get('bathrooms')
       
      
       latitude, longitude = airbnb.get_latlon_from_address(address)
       print('Latitude ', latitude)
       print('Longitude ', longitude)
       distance_from_center = airbnb.calculate_distance_from_center(latitude, longitude)
           
       
       # Case if empty
       if latitude == '': #not working right now
           return render_template("index_v2.html",
                                  latitude = latitude,
                                  my_form_result="Empty")
       else:
           price = airbnb.predict_price(model_pricing, latitude, longitude, entire_home, private_room, shared_room, has_wifi, has_kitchen, has_free_parking, accommodates, bathrooms, bedrooms, distance_from_center)
           price_lin = 10**(price)
           new_listing = [bathrooms, bedrooms, entire_home, private_room, shared_room, has_wifi, has_kitchen, has_free_parking, price_lin]
           av_ocmay, av_ocjun, av_ocjul, av_ocaug, av_rat = airbnb.descriptive_statistics(new_listing)           


           return render_template("index_v2.html",
                              my_price=price_lin,
                              av_may = av_ocmay,
                              av_june = av_ocjun,
                              av_july = av_ocjul,
                              av_aug = av_ocaug,
                              av_rating = av_rat,
                              my_form_result="NotEmpty")

       
# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True) #will run locally http://127.0.0.1:5000/
    #app.run(host="0.0.0.0", PORT=5000, debug=True) #when on google cloud
